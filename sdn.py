import math
import torch
from torch import Tensor, nn
from tqdm import trange
from junction import HouseholderScatteringJunction
from delay_filter import FractionalDelayLines
from position_utils import get_distances


class SDN(nn.Module):
    def __init__(self,
                 room_dim,
                 src_pos,
                 mic_pos,
                 N: int = 6,
                 sr: int = 16000,
                 c: float = 343.,
                 junction_type='householder',
                 delay_buffer_len: int = 512,
                 train_distances: bool = False,
                 max_distance_correction: float = 0.5,
                 distance_scaling: float = 10.0,
                 fir_order: int = 7,
                 alpha: float = 0.02,
                 device=None,
                 dtype=None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.device = device
        self.dtype = dtype
        # Factory kwargs used to move tensors to device and dtype consistently
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.N = N
        self.Nm1 = N - 1
        self.n_lines = self.N * self.Nm1
        self.delay_buffer_len = delay_buffer_len
        self.max_distance_correction = max_distance_correction
        self.distance_scaling = distance_scaling
        self.fir_order = fir_order
        self.G = c / sr  # Helper constant

        # Initialize junction filters (inverse sigmoid reparameterization for the scalar case)
        init_beta = -math.log((1 / math.sqrt(1 - alpha)) - 1) if self.fir_order == 0 else math.sqrt(1 - alpha)
        init_filt = torch.zeros(self.N, self.fir_order + 1, **self.factory_kwargs)
        init_filt[:, -1] = init_beta  
        self.junction_filters = nn.Parameter(init_filt)

        # Initialize junctions 
        if junction_type == 'householder':
            self.junctions = nn.ModuleList(
                [HouseholderScatteringJunction(self.N, j, **self.factory_kwargs) for j in range(self.N)]
            )
        else:
            raise ValueError(f'Junction type {junction_type} not recognized.')

        # Initialize permutation matrix P
        self.permutation_matrix = torch.zeros(self.n_lines, self.n_lines, **self.factory_kwargs)
        for i in range(1, self.n_lines + 1):
           f = (6 * i - ((i - 1) % self.N) - 1) % self.n_lines + 1
           self.permutation_matrix[i - 1, f - 1] = 1

        # Initialize pressure extraction weights
        self.pressure_weights = nn.Parameter(
            torch.full((self.n_lines,), 2 / self.Nm1, **self.factory_kwargs)
        )

        # Initialize distances
        dist_src_nodes, dist_nodes, dist_nodes_mic, dist_src_mic = get_distances(N, room_dim, src_pos, mic_pos)

        self.dist_src_nodes = torch.tensor(dist_src_nodes, **self.factory_kwargs)
        self.dist_nodes = torch.tensor(dist_nodes, **self.factory_kwargs)
        self.dist_nodes_mic = torch.tensor(dist_nodes_mic, **self.factory_kwargs)
        self.dist_src_mic = torch.tensor(dist_src_mic, **self.factory_kwargs)

        # Initialize unconstrained parameters for distance correction
        if train_distances:
            self.delta_src_nodes = nn.Parameter(
                torch.zeros_like(self.dist_src_nodes, **self.factory_kwargs) 
            )
            self.delta_nodes = nn.Parameter(
                torch.zeros_like(self.dist_nodes, **self.factory_kwargs) 
            )
            self.delta_nodes_mic = nn.Parameter(
                torch.zeros_like(self.dist_nodes_mic, **self.factory_kwargs) 
            )
            self.delta_src_mic = nn.Parameter(
                torch.zeros_like(self.dist_src_mic, **self.factory_kwargs)
            )
        else:
            self.register_buffer("delta_src_nodes", torch.zeros_like(self.dist_src_nodes, **self.factory_kwargs))
            self.register_buffer("delta_nodes", torch.zeros_like(self.dist_nodes, **self.factory_kwargs)) 
            self.register_buffer("delta_nodes_mic", torch.zeros_like(self.dist_nodes_mic, **self.factory_kwargs)) 
            self.register_buffer("delta_src_mic", torch.zeros_like(self.dist_src_mic, **self.factory_kwargs))        

    def reparametrize_delta(self, delta):
        """Return bounded distance correction term"""
        return self.max_distance_correction * torch.tanh(self.distance_scaling * delta)

    def forward(self, x: Tensor) -> Tensor:
        # Fix input shape
        if x.ndim == 1:
            x = x.unsqueeze(1)

        # Initialize output tensor
        y = torch.zeros_like(x)

        # Apply distance correction
        dist_src_nodes = self.dist_src_nodes + self.reparametrize_delta(self.delta_src_nodes)
        dist_nodes = self.dist_nodes + self.reparametrize_delta(self.delta_nodes)
        dist_nodes_mic = self.dist_nodes_mic + self.reparametrize_delta(self.delta_nodes_mic)
        dist_src_mic = self.dist_src_mic + self.reparametrize_delta(self.delta_src_mic)

        # Compute the attenuation coefficients from distances (spherical spreading law)
        source_gains = 0.5 * (self.G / dist_src_nodes).unsqueeze(1)
        mic_gains = 1 / (1 + dist_nodes_mic / dist_src_nodes).unsqueeze(1)
        direct_gain = 1 / dist_src_mic.squeeze()

        # Compute delays (in samples) from distances
        delay_src_nodes = dist_src_nodes / self.G
        delay_nodes = dist_nodes / self.G
        delay_nodes_mic = dist_nodes_mic / self.G
        delay_src_mic = dist_src_mic / self.G

        # Instantiate delay filters (inside the forward method to avoid retaining the graph across consecutive calls)
        src_to_mic = FractionalDelayLines(1, buffer_len=self.delay_buffer_len, **self.factory_kwargs)
        src_to_nodes = FractionalDelayLines(self.N, buffer_len=self.delay_buffer_len, **self.factory_kwargs)
        nodes_to_nodes = FractionalDelayLines(self.n_lines, buffer_len=self.delay_buffer_len, **self.factory_kwargs)
        nodes_to_mic = FractionalDelayLines(self.N, buffer_len=self.delay_buffer_len, **self.factory_kwargs)

        # Instantiate absorption/reflection filters (sigmoid reparameterization for the scalar case)
        junction_filters = torch.sigmoid(self.junction_filters) if self.fir_order == 0 else self.junction_filters
        junction_filters_nodes = junction_filters.repeat_interleave(self.Nm1, dim=0)

        # ========= Main simulation loop =========
        pp = 0.  # Initialize incident (incoming) waves to zero
        for k in trange(len(x)):
            # Add the (delayed) source signal to the global incident waves
            pp += src_to_nodes(source_gains * x[k], delay_src_nodes).repeat_interleave(self.Nm1, dim=0)

            # Compute the global reflected (outgoing) waves after local scattering
            pm = sum([junction(pp) for junction in self.junctions])

            # Compute the microphone signal (contribution from the nodes)
            y[k] = nodes_to_mic(mic_gains * (self.pressure_weights * pm).reshape(self.N, -1).sum(1, keepdim=True),
                                delay_nodes_mic, junction_filters).sum()

            # Add the (delayed) source signal to the microphone signal
            y[k] += direct_gain * src_to_mic(x[k], delay_src_mic).squeeze()

            # Compute the next global incident waves by delaying and permuting the global reflected waves
            pp = self.permutation_matrix @ nodes_to_nodes(pm, delay_nodes, junction_filters_nodes)

        return y

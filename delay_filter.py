import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FractionalDelayLines(nn.Module):
    """
    Differentiable Fractional Delay Lines.

    It allows to optimize delay values via backpropagation.

    [1] First introduced in A. I. Mezza, R. Giampiccolo, E. De Sena, and A. Bernardini, "Data-Driven Room Acoustic
    Modeling Via Differentiable Feedback Delay Networks With Learnable Delay Lines," EURASIP Journal of Audio, Speech,
    and Music Processing, vol. 2024, no. 1, pp. 1-20 (51), 2024.
    https://doi.org/10.1186/s13636-024-00371-5

    [2] The design of the fractional delay filter follows S.-C. Pei and Y.-C. Lai, "Closed Form Variable Fractional Time
    Delay Using FFT," in IEEE Signal Processing Letters, vol. 19, no. 5, pp. 299-302, 2012.
    https://doi.org/10.1109/LSP.2012.2191280
    """

    def __init__(self,
                 N: int,
                 buffer_len: int,
                 scale: float = 1.0,
                 device=None,
                 dtype=None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.factory_kwargs = {"device": device, "dtype": dtype}

        # Number of delay lines
        self.N = N

        # Delay scaling factor. Increasing it makes delays easier to update, even with small learning rates.
        self.scale = scale

        # Initialize the buffer
        self.buffer_len = buffer_len
        self.K = K = 2 * buffer_len
        self.buffer = torch.zeros(self.N, buffer_len, **self.factory_kwargs)

        # Pre-compute the frequencies of the closed-form fractional delay filter
        self.omega_left = (2 * torch.pi / K) * torch.arange(0, K // 2).unsqueeze(0).to(device)
        self.omega_right = (2 * torch.pi / K) * (K - torch.arange(K // 2 + 1, K).unsqueeze(0).to(device))


    def forward(self, inputs: Tensor, delays: Tensor, reflection_filters: Tensor = None) -> Tensor:
        # Fix input shape
        if inputs.ndim > 1:
            inputs = inputs.squeeze(1)

        # Circularly shift the buffer, overwrite the last position with zeros, and write the current sample
        self.buffer = torch.roll(self.buffer, -1, dims=-1)
        self.buffer[:, -1] = inputs

        # Apply zero padding
        buffer_pad = F.pad(self.buffer, (0, self.K - self.buffer_len))

        # Take the FFT
        fft_buffer = torch.fft.fft(buffer_pad, dim=-1)

        # Compute the fractional delay filter, see [2]
        delays = self.scale * delays.unsqueeze(1)
        H_left = torch.exp(-1j * self.omega_left * delays)
        H_mid = torch.cos(torch.pi * delays)
        H_right = torch.exp(1j * self.omega_right * delays)

        H = torch.cat([H_left, H_mid, H_right], dim=1)

        # Apply the filter in the frequency domain and go back in the time domain via IFFT
        delayed_buffer = torch.fft.ifft(H * fft_buffer)
        delayed_buffer = delayed_buffer.real.float()

        # Output the delayed sample
        if reflection_filters is not None:
            # Apply FIR wall absorption/reflection filters
            n_bins = reflection_filters.shape[-1] 
            outputs = delayed_buffer[:, self.buffer_len - n_bins - 1:self.buffer_len - 1]
            outputs = (reflection_filters * outputs).sum(-1)
        else:
            outputs = delayed_buffer[:, self.buffer_len - 1]
      
        return outputs.unsqueeze(1)

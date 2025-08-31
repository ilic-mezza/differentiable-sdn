import torch
from torch import Tensor, nn


class HouseholderScatteringJunction(nn.Module):
    def __init__(self,
                 N: int,  # Scattering matrix size
                 index: int,  # junction progressive number, 0 to 5
                 device=None,
                 dtype=None,
                 trainable: bool = True,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.N = N
        self.Nm1 = N - 1
        # Instantiate selection (routing) matrix
        self.R = torch.roll(torch.eye(self.Nm1, self.N * self.Nm1), index * self.Nm1).to(**factory_kwargs)
        # Instantiate an (N-1) identity matrix
        self.eye = torch.eye(self.Nm1).to(**factory_kwargs)

        if trainable:
            # Trainable admittance vector
            self.weight = nn.Parameter(
                torch.empty((1, self.Nm1), **factory_kwargs)
            )
        else:
            # Constant admittance vector -> Standard Householder matrix
            self.register_buffer('weight', torch.ones((self.Nm1, 1), **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    @property
    def S(self):
        # Ensure weights (admittances) are strictly positive
        weight = self.weight.abs() + 1e-12
        # Eq. (9) from the paper
        return (2 /weight.sum()) * weight.repeat(self.Nm1, 1) - self.eye

    def forward(self, x: Tensor) -> Tensor:
        # From right to left: global-to-local => scattering => local-to-global
        return  self.R.t() @ self.S @ self.R @ x

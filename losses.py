import torch
from torch import Tensor, nn
from curves import energy_decay_curve, echo_density_profile, mel_energy_decay_relief


def lp_error_fn(pred: Tensor, target: Tensor, power: float = 1.0, normalize: bool = False):
    """L^p loss with the option to normalize the output by the target norm"""
    loss = torch.mean(torch.abs(target - pred) ** power)
    if normalize:
        norm = torch.mean(torch.abs(target) ** power)
        return loss / norm
    return loss


class EDCLoss(nn.Module):
    def __init__(self, power: float = 2.0, **kwargs) -> None:
        """Normalized MSE between linear-amplitude full-band EDC curves"""
        super().__init__(**kwargs)
        self.power = power

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred = pred.squeeze()
        target = target.squeeze()
        edc_pred = energy_decay_curve(pred, return_db=False)
        edc_target = energy_decay_curve(target, return_db=False)
        loss = lp_error_fn(edc_pred, edc_target, self.power, normalize=True)
        print(f'EDC Loss: {loss.item():.6f}')
        return loss


class MelEDRLogLoss(nn.Module):
    def __init__(self, sr: int, power: float = 1.0, **kwargs) -> None:
        """Normalized MAE between log-amplitude mel-frequency EDRs"""
        super().__init__(**kwargs)
        self.sr = sr
        self.power = power

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred = pred.squeeze()
        target = target.squeeze()
        edr_pred = mel_energy_decay_relief(pred, self.sr, return_db=True)
        edr_target = mel_energy_decay_relief(target, self.sr, return_db=True)
        loss = lp_error_fn(edr_pred, edr_target, self.power, normalize=True)
        print(f'Mel-EDR log Loss: {loss.item():.6f}')
        return loss


class EDPLoss(nn.Module):
    def __init__(self, sr: int, win_duration: float = 0.02, power=2.0, **kwargs) -> None:
        """MSE between SoftEDP curves. For more details see https://doi.org/10.1186/s13636-024-00371-5"""
        super().__init__(**kwargs)
        self.sr = sr
        self.win_duration = win_duration
        self.power = power

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred = pred.squeeze()
        target = target.squeeze()
        profile_pred = echo_density_profile(pred, sr=self.sr, win_duration=self.win_duration, differentiable=True)
        profile_target = echo_density_profile(target, sr=self.sr, win_duration=self.win_duration, differentiable=True)
        loss = lp_error_fn(profile_pred, profile_target, self.power, normalize=False)
        print(f'EDP Loss: {loss.item():.6f}')
        return loss
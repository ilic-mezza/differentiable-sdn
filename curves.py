import torch
from torch import Tensor
import torchaudio.transforms as T
from typing import Union, Tuple


def energy_decay_curve(h: Tensor, return_db: bool = False) -> Tensor:
    """Schroeder’s full-band EDC (https://ccrma.stanford.edu/~jos/pasp/Energy_Decay_Curve.html)"""

    h_pow = h ** 2
    cum_sum = torch.cumsum(h_pow, dim=0)
    edc = h_pow + cum_sum[-1] - cum_sum
    return 10 * torch.log10(edc) if return_db else edc


def mel_energy_decay_relief(h: Tensor, sr: int, return_db: bool = True) -> Tensor:
    """Mel-frequency EDR (https://ccrma.stanford.edu/~jos/pasp/Energy_Decay_Relief.html)"""

    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=512,
        win_length=320,
        hop_length=160,
        f_min=0.0,
        f_max=sr/2,
        n_mels=64,
        power=2.0,
        window_fn=torch.hann_window,
        normalized=False,
    ).to(h.device)
    spec = mel_spectrogram(h)
    cum_sum = torch.cumsum(spec, dim=-1)
    edr = spec + cum_sum[:, -1:] - cum_sum
    return T.AmplitudeToDB(top_db=80)(edr) if return_db else edr


def echo_density_profile(h: Tensor, sr: int, win_duration: float = 0.02, differentiable: bool = False,
                         kappa: Union[float, Tuple[float, float]] = (1e2, 1e5)) -> Tensor:
    """
    Echo Density Profile (EDP)

    Based on J. S. Abel & P. Huang, "A Simple, Robust Measure of Reverberation Echo Density," 121st AES Convention, 2006
    https://ccrma.stanford.edu/courses/318/mini-courses/rooms/mus318_Abel_Lecture/echo%20density.pdf

    Differentiable approximation (SoftEDP) from A. I. Mezza, R. Giampiccolo, E. De Sena, and A. Bernardini, "Data-Driven
    Room Acoustic Modeling Via Differentiable Feedback Delay Networks With Learnable Delay Lines," EURASIP Journal of
    Audio, Speech, and Music Processing, vol. 2024, no. 1, pp. 1-20 (51), 2024.
    https://doi.org/10.1186/s13636-024-00371-5
    """

    # Instantiate the odd-length symmetric Hann window
    win_len = int(win_duration * sr)
    win_len = win_len + 1 if win_len % 2 == 0 else win_len
    win = torch.hann_window(win_len, periodic=False, device=h.device)
    win = win / win.sum()  # sum to 1

    # Initialize the echo density profile
    profile = torch.zeros(len(h) - win_len)

    # Define the sequence of Sigmoid scaling factors
    if isinstance(kappa, tuple):
        # linearly increasing scaling factors
        kappa = torch.linspace(kappa[0], kappa[1], len(profile))
    else:
        # constant scaling
        kappa = [kappa] * len(profile)

    # Process each window sliding with unit stride
    for i in range(len(profile)):
        h_frame = h[i:i + win_len].abs()

        sigma = torch.sqrt(torch.sum(win * (h_frame ** 2)))

        if differentiable:
            indicator = torch.sigmoid(kappa[i] * (h_frame - sigma))  # SoftEDP
        else:
            indicator = (h_frame > sigma).float()  # Original non-differentiable EDP

        profile[i] = torch.sum(win * indicator)

    # 1/erfc(1 /√2) ≈ 3.15148718753
    profile = 3.15148718753 * profile

    return profile


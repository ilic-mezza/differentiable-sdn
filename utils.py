import torch
import random
import numpy as np
import torchaudio.functional as F
import pyroomacoustics as pra
from scipy.io import wavfile


def seed_everything(seed):
    """Set the random seed across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_homula_rir(rir_path, ula_index: int, sr: int, trim: bool=False):
    """
    Load and preprocess a single Room Impulse Response (RIR) from HOMULA-RIR.
    https://doi.org/10.1109/ICASSPW62465.2024.10626753

    This function reads a multichannel RIR WAV file recorded with a Uniform Linear Array (ULA),
    selects a specific microphone channel, converts it to a tensor, resamples it to the target
    sampling rate, crops it to the effective reverberation time, and normalizes it to unit norm.
    """
    # Read WAV file
    orig_sr, ula_rir = wavfile.read(rir_path)

    # Select the target microphone from the ULA
    rir = ula_rir[:, ula_index]

    # Convert to tensor
    rir = torch.tensor(rir[None, :])

    # Resample RIR
    rir = F.resample(rir, orig_sr, sr)

    # Crop RIR to the reverberation time
    if trim:
        reverb_time = pra.experimental.rt60.measure_rt60(rir[0], fs=sr, decay_db=30)
        rir = rir[:, :int(reverb_time * sr)]

    # Normalize RIR to unit norm
    rir /= rir.norm()

    return rir
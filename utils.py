"""
Audio read/write and manipulation utilities.

Includes read/write helpers, channel normalization, and an insert function to
place a voice mail audio clip into an original file at a given drop time.
"""

import os
import soundfile as sf
import numpy as np


def read_audio(path):
    """
    Read audio from path and return samples and sample rate.

    Ensures returned audio is float32.
    """
    audio, sr = sf.read(path)
    if audio.dtype != np.float32:
        audio = audio.astype('float32')
    return audio, sr


def write_audio(path, audio, sr):
    """
    Write audio to path, creating parent directories if needed, and clamp
    samples to [-1, 1] before writing.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    max_val = np.abs(audio).max() if audio.size else 0.0
    if max_val > 1.0:
        audio = audio / max_val
    sf.write(path, audio, sr)


def ensure_channels(audio, target_channels):
    """
    Ensure audio has the requested number of channels.

    Args:
        audio (np.ndarray): (N,) mono or (N, C) multi-channel array.
        target_channels (int): 1 or 2 are commonly used.

    Returns:
        np.ndarray: audio with the requested channels.
    """
    if audio.ndim == 1 and target_channels == 1:
        return audio
    if audio.ndim == 1 and target_channels == 2:
        return np.stack([audio, audio], axis=1)
    if audio.ndim == 2 and target_channels == 1:
        return np.mean(audio, axis=1)
    if audio.ndim == 2 and audio.shape[1] == target_channels:
        return audio
    mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio
    if target_channels == 1:
        return mono
    return np.stack([mono, mono], axis=1)


def insert_voice_mail_at_drop(original_path, voice_mail_path, drop_time, output_path):
    """
    Insert voice_mail audio into original audio at drop_time (seconds) and save.

    If drop_time is beyond the original length, the voice mail is appended.
    Sample rates are aligned via simple linear resampling if needed, and
    channels are matched to the original file.
    """
    orig, sr_orig = read_audio(original_path)
    vm, sr_vm = read_audio(voice_mail_path)

    if sr_vm != sr_orig:
        print(f"Resampling voice mail from {sr_vm}Hz to {sr_orig}Hz")
        import math
        vm_len = vm.shape[0]
        new_len = int(math.ceil(vm_len * (sr_orig / sr_vm)))
        if vm.ndim == 1:
            xp = np.linspace(0, 1, vm_len)
            x = np.linspace(0, 1, new_len)
            vm = np.interp(x, xp, vm).astype('float32')
        else:
            channels = []
            for c in range(vm.shape[1]):
                xp = np.linspace(0, 1, vm_len)
                x = np.linspace(0, 1, new_len)
                ch = np.interp(x, xp, vm[:, c])
                channels.append(ch)
            vm = np.stack(channels, axis=1).astype('float32')

    orig_channels = 1 if orig.ndim == 1 else orig.shape[1]
    vm = ensure_channels(vm, orig_channels)

    idx = int(round(drop_time * sr_orig)) if drop_time >= 0 else 0
    idx = max(0, min(idx, orig.shape[0]))

    new_audio = np.concatenate([orig[:idx], vm, orig[idx:]], axis=0)

    write_audio(output_path, new_audio, sr_orig)
    return output_path

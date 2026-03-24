import numpy as np


def smooth_audio_edges(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    fade_ms = 12
    pad_ms = 4
    fade_samples = max(1, int(sample_rate * fade_ms / 1000))
    pad_samples = max(1, int(sample_rate * pad_ms / 1000))

    if audio.size < 2:
        return audio

    fade_len = min(fade_samples, audio.size // 2)
    if fade_len > 0:
        ramp = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        audio[:fade_len] *= ramp
        audio[-fade_len:] *= ramp[::-1]

    return np.concatenate(
        (
            np.zeros(pad_samples, dtype=np.float32),
            audio,
            np.zeros(pad_samples, dtype=np.float32),
        )
    )


def normalize_audio(audio: np.ndarray, gain: float) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        return audio / peak * gain
    return audio


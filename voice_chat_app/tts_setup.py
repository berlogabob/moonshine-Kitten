import os

import numpy as np
from kittentts import KittenTTS

from .config import LEGACY_VOICE_ALIASES


def resolve_voice(requested_voice: str, voices_file: str):
    if not os.path.isfile(voices_file):
        raise FileNotFoundError(
            f"Voices file not found: {voices_file}\n"
            "Re-download:\n"
            "  rm -rf ./kitten-tts-mini-0.8\n"
            "  uv run hf download KittenML/kitten-tts-mini-0.8 --local-dir ./kitten-tts-mini-0.8"
        )

    with np.load(voices_file) as voice_archive:
        available_voices = list(voice_archive.files)

    if not available_voices:
        raise ValueError(f"No voices found in archive: {voices_file}")

    if requested_voice in available_voices:
        return requested_voice, available_voices, None

    for candidate in LEGACY_VOICE_ALIASES.get(requested_voice, ()):
        if candidate in available_voices:
            return (
                candidate,
                available_voices,
                f"Configured voice '{requested_voice}' not found; using compatible voice '{candidate}'.",
            )

    fallback_voice = available_voices[0]
    return (
        fallback_voice,
        available_voices,
        f"Configured voice '{requested_voice}' not found; using first available voice '{fallback_voice}'.",
    )


def load_compatible_voices(voices_file: str, session):
    if not os.path.isfile(voices_file):
        raise FileNotFoundError(
            f"Voices file not found: {voices_file}\n"
            "Re-download:\n"
            "  rm -rf ./kitten-tts-mini-0.8\n"
            "  uv run hf download KittenML/kitten-tts-mini-0.8 --local-dir ./kitten-tts-mini-0.8"
        )

    style_input = next(inp for inp in session.get_inputs() if inp.name == "style")
    expected_dim = style_input.shape[-1]
    if not isinstance(expected_dim, int):
        raise ValueError(f"Unexpected ONNX style input shape: {style_input.shape}")

    compatible_voices = {}
    with np.load(voices_file) as voice_archive:
        for voice_name in voice_archive.files:
            style = np.asarray(voice_archive[voice_name], dtype=np.float32)
            if style.ndim == 2:
                style = style.mean(axis=0)
            if style.ndim != 1:
                raise ValueError(
                    f"Unsupported style tensor shape for voice '{voice_name}': {style.shape}"
                )
            if style.shape[0] != expected_dim:
                raise ValueError(
                    f"Voice '{voice_name}' style width {style.shape[0]} does not match model input width {expected_dim}."
                )
            compatible_voices[voice_name] = style.reshape(1, expected_dim)

    if not compatible_voices:
        raise ValueError(f"No voices found in archive: {voices_file}")

    return compatible_voices


def build_tts(local_model_file: str, local_voices_file: str, requested_voice: str):
    print(f"Loading TTS model from: {local_model_file}")
    if not os.path.isfile(local_model_file):
        raise FileNotFoundError(
            f"Model file not found: {local_model_file}\n"
            "Download command:\n"
            "  uv run hf download KittenML/kitten-tts-mini-0.8 --local-dir ./kitten-tts-mini-0.8"
        )

    file_size_mb = os.path.getsize(local_model_file) / (1024 * 1024)
    if file_size_mb < 50:
        raise ValueError(
            f"Model file appears corrupted (only {file_size_mb:.1f} MB instead of ~75 MB)\n"
            "Re-download:\n"
            "  rm -rf ./kitten-tts-mini-0.8\n"
            "  uv run hf download KittenML/kitten-tts-mini-0.8 --local-dir ./kitten-tts-mini-0.8"
        )

    tts = KittenTTS(local_model_file)
    tts._voices = load_compatible_voices(local_voices_file, tts._session)
    selected_voice, available_voices, voice_warning = resolve_voice(
        requested_voice, local_voices_file
    )
    if voice_warning:
        print(f"Voice selection warning: {voice_warning}")
    print(f"Available voices: {', '.join(available_voices)}")
    print(f"Using TTS voice: {selected_voice}")
    print("TTS model loaded successfully")

    return tts, selected_voice, available_voices


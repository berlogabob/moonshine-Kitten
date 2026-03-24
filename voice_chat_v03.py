import os  # File existence and size checks
import threading  # Needed for background microphone listening thread
import time  # For sleep delays and cooldowns
import warnings  # To suppress noisy runtime warnings

import numpy as np  # Array handling for audio data
import ollama  # Local LLM client (Gemma, Phi3, etc.)
import sounddevice as sd  # Audio playback and (potential) recording
from kittentts import KittenTTS
from moonshine_voice import MicTranscriber, ModelArch, TranscriptEventListener

# Fix phonemizer / espeak-ng lookup on macOS (Homebrew path)
from phonemizer.backend.espeak.wrapper import EspeakWrapper

EspeakWrapper.set_library("/opt/homebrew/lib/libespeak-ng.dylib")

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ===================== CONFIGURATION =====================

OLLAMA_MODEL = "gemma2:2b"  # Change to "phi3:mini", "tinyllama:1.1b", etc. if desired
VOICE = "Jasper"  # TTS voice — try "Bella" or "Luna" if Jasper sounds weak
MODEL_PATH = "/Users/berloga/Library/Caches/moonshine_voice/download.moonshine.ai/model/medium-streaming-en/quantized"
MODEL_ARCH = 5  # Architecture index for Moonshine model
AUDIO_SAMPLE_RATE = 24000

# Global flag to prevent feedback loop (AI hearing itself)
is_speaking = False

# System prompt that controls LLM behavior
SYSTEM_PROMPT = """
You are a short, concise voice assistant.
Answer in 1-2 short sentences maximum.
NEVER respond to your own previous messages.
Never repeat yourself.
No emojis, markdown, asterisks, lists, "Thinking".
Direct and useful answer only.
"""

# Path to the locally downloaded ONNX model file
LOCAL_MODEL_FILE = "./kitten-tts-mini-0.8/kitten_tts_mini_v0_8.onnx"
LOCAL_VOICES_FILE = "./kitten-tts-mini-0.8/voices.npz"

LEGACY_VOICE_ALIASES = {
    "Jasper": ("expr-voice-2-m", "expr-voice-3-m", "expr-voice-4-m", "expr-voice-5-m"),
    "Bella": ("expr-voice-2-f", "expr-voice-3-f", "expr-voice-4-f", "expr-voice-5-f"),
    "Luna": ("expr-voice-3-f", "expr-voice-4-f", "expr-voice-5-f", "expr-voice-2-f"),
}


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

# ===================== LOAD TTS MODEL =====================
try:
    print(f"Loading TTS model from: {LOCAL_MODEL_FILE}")

    # Basic file existence check
    if not os.path.isfile(LOCAL_MODEL_FILE):
        raise FileNotFoundError(
            f"Model file not found: {LOCAL_MODEL_FILE}\n"
            "Download command:\n"
            "  uv run hf download KittenML/kitten-tts-mini-0.8 --local-dir ./kitten-tts-mini-0.8"
        )

    # Check file size to detect incomplete/corrupted downloads
    file_size_mb = os.path.getsize(LOCAL_MODEL_FILE) / (1024 * 1024)
    if file_size_mb < 50:
        raise ValueError(
            f"Model file appears corrupted (only {file_size_mb:.1f} MB instead of ~75 MB)\n"
            "Re-download:\n"
            "  rm -rf ./kitten-tts-mini-0.8\n"
            "  uv run hf download KittenML/kitten-tts-mini-0.8 --local-dir ./kitten-tts-mini-0.8"
        )

    # Load ONNX model and adapt local style embeddings to model's expected shape.
    tts = KittenTTS(LOCAL_MODEL_FILE)
    tts._voices = load_compatible_voices(LOCAL_VOICES_FILE, tts._session)
    SELECTED_VOICE, AVAILABLE_VOICES, voice_warning = resolve_voice(
        VOICE, LOCAL_VOICES_FILE
    )
    if voice_warning:
        print(f"Voice selection warning: {voice_warning}")
    print(f"Available voices: {', '.join(AVAILABLE_VOICES)}")
    print(f"Using TTS voice: {SELECTED_VOICE}")
    print("TTS model loaded successfully")

except Exception as e:
    print(f"Failed to load TTS model: {e}")
    print("Quick troubleshooting steps:")
    print(f"  ls -lh {LOCAL_MODEL_FILE}")
    print(
        "  rm -rf ./kitten-tts-mini-0.8 && uv run hf download KittenML/kitten-tts-mini-0.8 --local-dir ./kitten-tts-mini-0.8"
    )
    raise

# Chat history for context (limited to last 10 messages)
chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]


def clean_text(text: str) -> str:
    """Strip unwanted markdown, emojis, and extra whitespace from LLM output."""
    text = text.replace("**", "").replace("*", "").replace("`", "")
    text = (
        text.replace("🤖", "").replace("👋", "").replace("😊", "").replace("...", " ")
    )
    return " ".join(text.split()).strip()


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

    # Zero padding prevents DAC boundary discontinuities between utterances.
    return np.concatenate(
        (
            np.zeros(pad_samples, dtype=np.float32),
            audio,
            np.zeros(pad_samples, dtype=np.float32),
        )
    )


# Listener that gets called every time Moonshine finishes transcribing a full sentence
class SafeListener(TranscriptEventListener):
    def on_line_completed(self, event):
        global is_speaking

        # Skip processing if the AI is currently speaking (prevents echo loop)
        if is_speaking:
            return

        user_text = event.line.text.strip()
        if not user_text or len(user_text) < 3:
            return

        print(f"\nYou: {user_text}")
        chat_history.append({"role": "user", "content": user_text})

        # Keep history short to save memory and tokens
        if len(chat_history) > 10:
            chat_history[:] = chat_history[-10:]

        try:
            # Get response from local LLM via ollama
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=chat_history,
                options={
                    "temperature": 0.5,  # Lower = more deterministic
                    "num_predict": 80,  # Max tokens in reply
                    "repeat_penalty": 1.25,  # Discourage repetition
                },
            )["message"]["content"].strip()

            clean_resp = clean_text(response)
            print(f"AI: {clean_resp}")
            chat_history.append({"role": "assistant", "content": clean_resp})

            # Generate and play speech
            is_speaking = True
            raw = tts.generate(clean_resp, voice=SELECTED_VOICE)
            audio = np.asarray(raw, dtype=np.float32)

            # Normalize volume and reduce it to minimize microphone feedback
            peak = np.max(np.abs(audio))
            if peak > 1e-6:
                audio = audio / peak * 0.72
            audio = smooth_audio_edges(audio, AUDIO_SAMPLE_RATE)

            print("▶️ Speaking...")
            sd.play(audio, samplerate=AUDIO_SAMPLE_RATE)
            sd.wait()

            # Short cooldown after speech to let the microphone settle
            time.sleep(0.8)
            is_speaking = False

        except Exception as e:
            print(f"Processing error: {e}")
            is_speaking = False


# ===================== STARTUP & MAIN LOOP =====================
print("Voice chat started — feedback loop protection enabled")
print(f"  LLM model : {OLLAMA_MODEL}")
print(f"  TTS voice : {SELECTED_VOICE}")
print(f"  TTS model : {LOCAL_MODEL_FILE}")
print("  AI should NOT talk to itself\n")

# Quick test to make sure TTS audio works
print("TTS test...")
try:
    test_audio = np.asarray(
        tts.generate("Test sound one two three", voice=SELECTED_VOICE), dtype=np.float32
    )
    peak = np.max(np.abs(test_audio))
    if peak > 1e-6:
        test_audio = test_audio / peak * 0.8
    test_audio = smooth_audio_edges(test_audio, AUDIO_SAMPLE_RATE)
    sd.play(test_audio, samplerate=AUDIO_SAMPLE_RATE)
    sd.wait()
    print("TTS test completed.\n")
except Exception as e:
    print(f"TTS test failed: {e}")

# Start real-time microphone transcription
try:
    print("Starting microphone listener...")
    transcriber = MicTranscriber(
        model_path=MODEL_PATH, model_arch=ModelArch(MODEL_ARCH)
    )
    transcriber.add_listener(SafeListener())
    transcriber.start()
    threading.Event().wait()  # Keeps the script alive until Ctrl+C

except KeyboardInterrupt:
    print("\nStopping...")
    if "transcriber" in locals():
        transcriber.stop()
    print("Goodbye!")

except Exception as e:
    print(f"Launch error: {e}")

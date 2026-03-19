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

    # Actually load the model into memory
    tts = KittenTTS(LOCAL_MODEL_FILE)
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
            raw = tts.generate(clean_resp, voice=VOICE)
            audio = np.asarray(raw, dtype=np.float32)

            # Normalize volume and reduce it to minimize microphone feedback
            peak = np.max(np.abs(audio))
            if peak > 1e-6:
                audio = audio / peak * 0.72

            print("▶️ Speaking...")
            sd.play(audio, samplerate=24000)
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
print(f"  TTS voice : {VOICE}")
print(f"  TTS model : {LOCAL_MODEL_FILE}")
print("  AI should NOT talk to itself\n")

# Quick test to make sure TTS audio works
print("TTS test...")
try:
    test_audio = np.asarray(
        tts.generate("Test sound one two three", voice=VOICE), dtype=np.float32
    )
    peak = np.max(np.abs(test_audio))
    if peak > 1e-6:
        test_audio = test_audio / peak * 0.8
    sd.play(test_audio, samplerate=24000)
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

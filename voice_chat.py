import queue
import threading
import warnings

import numpy as np
import ollama
import sounddevice as sd
from kittentts import KittenTTS
from moonshine_voice import MicTranscriber, ModelArch, TranscriptEventListener, download

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ──────────────────────────────────────────────
# SETTINGS
# ──────────────────────────────────────────────

OLLAMA_MODEL = "gemma2:2b"
VOICE = "Jasper"  # try "Bella" if Jasper silent

# Auto-download model if not present
MODEL_PATH, MODEL_ARCH = download.get_model_for_language(
    "en", ModelArch.MEDIUM_STREAMING
)

SYSTEM_PROMPT = """
You are a short voice assistant.
Answer in 1–2 short sentences or one word.
No repetition, no extra questions unless asked.
No emojis, markdown, asterisks, lists, "Thinking".
Direct useful answer only.
"""

# ──────────────────────────────────────────────
# INIT
# ──────────────────────────────────────────────

tts = KittenTTS("KittenML/kitten-tts-mini-0.8")

chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]

# Queue for TTS playback to avoid blocking
tts_queue = queue.Queue()


def clean_text(text: str) -> str:
    text = text.replace("**", "").replace("*", "").replace("`", "")
    text = (
        text.replace("🤖", "").replace("👋", "").replace("😊", "").replace("...", " ")
    )
    return " ".join(text.split()).strip()


def play_tts_async(audio_data: np.ndarray):
    """Play TTS audio in a non-blocking way."""

    def _play():
        try:
            sd.play(audio_data, samplerate=24000)
            sd.wait()
        except Exception as e:
            print(f"TTS playback error: {e}")

    thread = threading.Thread(target=_play, daemon=True)
    thread.start()


class VADListener(TranscriptEventListener):
    """Voice Activity Detection listener - processes all speech automatically."""

    def __init__(self):
        super().__init__()
        self.last_speech_time = 0

    def on_line_completed(self, event):
        user_text = event.line.text.strip()
        if not user_text or len(user_text) < 3:
            return

        print(f"\nYou: {user_text}")
        chat_history.append({"role": "user", "content": user_text})

        if len(chat_history) > 10:
            chat_history[:] = chat_history[-10:]

        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=chat_history,
                options={
                    "temperature": 0.55,
                    "num_predict": 90,
                    "repeat_penalty": 1.18,
                },
            )["message"]["content"].strip()

            clean_resp = clean_text(response)
            print(f"AI: {clean_resp}")

            chat_history.append({"role": "assistant", "content": clean_resp})

            # Generate and play TTS asynchronously
            raw_audio = tts.generate(clean_resp, voice=VOICE)
            audio_out = np.asarray(raw_audio, dtype=np.float32)
            peak = np.max(np.abs(audio_out))
            if peak > 1e-6:
                audio_out = audio_out / peak * 0.98

            play_tts_async(audio_out)

        except Exception as e:
            print(f"Processing error: {e}")


# ──────────────────────────────────────────────
# START (Continuous VAD Mode)
# ──────────────────────────────────────────────

print("Voice chat started (VAD mode - speak naturally)")
print(f"   LLM: {OLLAMA_MODEL}")
print(f"   TTS voice: {VOICE}")
print("   Speak naturally - I'll respond automatically")
print("   Press Ctrl+C to exit\n")

# TTS test
print("TTS test...")
test_raw = tts.generate("Test sound one two three", voice=VOICE)
test_audio = np.asarray(test_raw, dtype=np.float32)
peak = np.max(np.abs(test_audio))
if peak > 1e-6:
    test_audio = test_audio / peak * 0.98
sd.play(test_audio, samplerate=24000)
sd.wait()
print("TTS test done. Heard sound?\n")

# Start transcriber with VAD listener
transcriber = MicTranscriber(model_path=MODEL_PATH, model_arch=ModelArch(MODEL_ARCH))
transcriber.add_listener(VADListener())
transcriber.start()

print("Listening... (Ctrl+C to stop)\n")

try:
    while True:
        threading.Event().wait(1)  # Keep main thread alive
except KeyboardInterrupt:
    print("\nStopping...")

transcriber.stop()
print("Goodbye!")

import threading
import time
import warnings

import numpy as np
import ollama
import sounddevice as sd
from kittentts import KittenTTS
from moonshine_voice import MicTranscriber, ModelArch, TranscriptEventListener

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ===================== НАСТРОЙКИ =====================
OLLAMA_MODEL = "gemma2:2b"  # или phi3:mini / tinyllama:1.1b
VOICE = "Jasper"  # Bella / Luna — попробуй, если Jasper тихий
MODEL_PATH = "/Users/berloga/Library/Caches/moonshine_voice/download.moonshine.ai/model/medium-streaming-en/quantized"
MODEL_ARCH = 5

# ЗАЩИТА ОТ FEEDBACK LOOP
is_speaking = False  # главный флаг — пока говорит, не слушаем

SYSTEM_PROMPT = """
You are a short voice assistant.
Answer in 1-2 short sentences maximum.
NEVER respond to your own previous messages.
Never repeat yourself.
No emojis, markdown, asterisks, lists, "Thinking".
Direct useful answer only.
"""

tts = KittenTTS("KittenML/kitten-tts-mini-0.8")

chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]


def clean_text(text: str) -> str:
    text = text.replace("**", "").replace("*", "").replace("`", "")
    text = (
        text.replace("🤖", "").replace("👋", "").replace("😊", "").replace("...", " ")
    )
    return " ".join(text.split()).strip()


class SafeListener(TranscriptEventListener):
    def on_line_completed(self, event):
        global is_speaking
        if is_speaking:  # ← ЗАЩИТА ОТ САМОГО СЕБЯ
            return

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
                options={"temperature": 0.5, "num_predict": 80, "repeat_penalty": 1.25},
            )["message"]["content"].strip()

            clean_resp = clean_text(response)
            print(f"AI: {clean_resp}")

            chat_history.append({"role": "assistant", "content": clean_resp})

            # === TTS с защитой ===
            is_speaking = True

            raw = tts.generate(clean_resp, voice=VOICE)
            audio = np.asarray(raw, dtype=np.float32)

            # понижаем громкость, чтобы микрофон меньше ловил
            peak = np.max(np.abs(audio))
            if peak > 1e-6:
                audio = audio / peak * 0.72  # было 0.98 — теперь тише

            print("▶️ Speaking...")
            sd.play(audio, samplerate=24000)
            sd.wait()

            # cooldown — даём микрофону "успокоиться"
            time.sleep(0.8)
            is_speaking = False

        except Exception as e:
            print(f"Error: {e}")
            is_speaking = False


# ===================== ЗАПУСК =====================
print("Voice chat started — FIXED feedback loop")
print(f"   Model: {OLLAMA_MODEL}")
print(f"   TTS: {VOICE}")
print("   Now it should NOT talk to itself\n")

# Тест звука
print("TTS test...")
test_audio = np.asarray(
    tts.generate("Test sound one two three", voice=VOICE), dtype=np.float32
)
peak = np.max(np.abs(test_audio))
if peak > 1e-6:
    test_audio = test_audio / peak * 0.8
sd.play(test_audio, samplerate=24000)
sd.wait()
print("TTS test done.\n")

try:
    transcriber = MicTranscriber(
        model_path=MODEL_PATH, model_arch=ModelArch(MODEL_ARCH)
    )
    transcriber.add_listener(SafeListener())
    transcriber.start()

    threading.Event().wait()

except KeyboardInterrupt:
    print("\nStopping...")
    if "transcriber" in locals():
        transcriber.stop()
    print("Goodbye!")
except Exception as e:
    print(f"Launch error: {e}")

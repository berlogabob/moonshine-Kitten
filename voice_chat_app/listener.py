import time

import numpy as np
import ollama
import sounddevice as sd
from moonshine_voice import TranscriptEventListener

from .audio import normalize_audio, smooth_audio_edges
from .ollama_settings import EMPTY_REPLY_FALLBACK, OLLAMA_OPTIONS
from .text_utils import clean_text


class SafeListener(TranscriptEventListener):
    def __init__(
        self,
        *,
        chat_history,
        tts,
        selected_voice: str,
        ollama_model: str,
        audio_sample_rate: int,
    ):
        super().__init__()
        self.chat_history = chat_history
        self.tts = tts
        self.selected_voice = selected_voice
        self.ollama_model = ollama_model
        self.audio_sample_rate = audio_sample_rate
        self.is_speaking = False

    def on_line_completed(self, event):
        if self.is_speaking:
            return

        user_text = event.line.text.strip()
        if not user_text or len(user_text) < 3:
            return

        print(f"\nYou: {user_text}")
        self.chat_history.append({"role": "user", "content": user_text})
        if len(self.chat_history) > 10:
            self.chat_history[:] = self.chat_history[-10:]

        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=self.chat_history,
                options=OLLAMA_OPTIONS,
            )

            message = response.get("message") or {}
            content = message.get("content")
            if content is None:
                content = ""
            reply_text = content.strip()
            if not reply_text:
                print("AI returned empty content, using fallback reply.")
                reply_text = EMPTY_REPLY_FALLBACK

            clean_resp = clean_text(reply_text)
            if not clean_resp:
                clean_resp = EMPTY_REPLY_FALLBACK

            print(f"AI: {clean_resp}")
            self.chat_history.append({"role": "assistant", "content": clean_resp})

            self.is_speaking = True
            raw = self.tts.generate(clean_resp, voice=self.selected_voice)
            audio = np.asarray(raw, dtype=np.float32)
            audio = normalize_audio(audio, gain=0.72)
            audio = smooth_audio_edges(audio, self.audio_sample_rate)

            print("▶️ Speaking...")
            sd.play(audio, samplerate=self.audio_sample_rate)
            sd.wait()

            time.sleep(0.8)
            self.is_speaking = False
        except Exception as e:
            print(f"Processing error: {e}")
            self.is_speaking = False

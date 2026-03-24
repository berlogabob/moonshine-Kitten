from .llm import generate_assistant_reply
from .speech_pipeline import speak_text_chunked
from .state import ChatState


class VoiceChatEngine:
    def __init__(
        self,
        *,
        tts,
        selected_voice: str,
        ollama_model: str,
        audio_sample_rate: int,
        cooldown_seconds: float = 0.8,
    ):
        self.tts = tts
        self.selected_voice = selected_voice
        self.ollama_model = ollama_model
        self.audio_sample_rate = audio_sample_rate
        self.cooldown_seconds = cooldown_seconds
        self.chat = ChatState(max_messages=10)
        self.is_speaking = False

    def process_user_text(self, user_text: str):
        if self.is_speaking:
            return

        user_text = user_text.strip()
        if not user_text or len(user_text) < 3:
            return

        print(f"\nYou: {user_text}")
        self.chat.add_user(user_text)

        reply = generate_assistant_reply(self.chat.history, model=self.ollama_model)
        print(f"AI: {reply}")
        self.chat.add_assistant(reply)

        self.is_speaking = True
        print("▶️ Speaking...")
        speak_text_chunked(
            tts=self.tts,
            text=reply,
            voice=self.selected_voice,
            sample_rate=self.audio_sample_rate,
            gain=0.72,
            max_chunk_chars=90,
        )


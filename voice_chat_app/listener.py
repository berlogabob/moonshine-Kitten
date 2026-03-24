import time

from moonshine_voice import TranscriptEventListener

from .engine import VoiceChatEngine


class SafeListener(TranscriptEventListener):
    def __init__(
        self,
        *,
        engine: VoiceChatEngine,
    ):
        super().__init__()
        self.engine = engine

    def on_line_completed(self, event):
        if self.engine.is_speaking:
            return

        try:
            self.engine.process_user_text(event.line.text)
            if self.engine.is_speaking:
                time.sleep(self.engine.cooldown_seconds)
        except Exception as e:
            print(f"Processing error: {e}")
        finally:
            self.engine.is_speaking = False

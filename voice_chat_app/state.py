from .prompts import SYSTEM_PROMPT


class ChatState:
    def __init__(self, max_messages: int = 10):
        self.max_messages = max_messages
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def add_user(self, text: str):
        self.history.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str):
        self.history.append({"role": "assistant", "content": text})
        self._trim()

    def _trim(self):
        if len(self.history) > self.max_messages:
            self.history[:] = self.history[-self.max_messages :]


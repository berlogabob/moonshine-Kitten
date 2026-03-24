# Copilot Instructions for moonshine-Kitten

## Build, test, and lint commands

This repository is a `uv`-managed Python project (see `pyproject.toml`).

```bash
# install dependencies into .venv
uv sync

# one-time: download local KittenTTS model files
uv run hf download KittenML/kitten-tts-mini-0.8 --local-dir ./kitten-tts-mini-0.8

# run the current main app
uv run voice_chat_v03.py

# run earlier variants
uv run voice_chat.py
uv run voice_chat_v02.py
```

System dependency used by KittenTTS phonemizer:

```bash
# macOS
brew install espeak-ng

# Ubuntu/Debian
sudo apt update && sudo apt install espeak-ng libespeak-ng1
```

There is no configured lint/test command or test suite in this repo yet (no `tests/`, `pytest`, `ruff`, or lint scripts).

## High-level architecture

The app is a local real-time voice loop:

1. **STT input**: `moonshine_voice.MicTranscriber` streams microphone transcription and emits completed-line events.
2. **LLM response**: each completed user line is appended to `chat_history` and sent to `ollama.chat(...)`.
3. **Response cleanup**: output text is sanitized (`clean_text`) to remove markdown/emojis and collapse whitespace.
4. **TTS output**: `KittenTTS.generate(...)` creates waveform audio, converted to `numpy.float32`.
5. **Playback**: audio is peak-normalized and played through `sounddevice` at 24kHz.

Main implementation variants:

- `voice_chat.py`: startup checks for local ONNX model file and size, explicit `EspeakWrapper` path, synchronous TTS playback with feedback guard.
- `voice_chat_v02.py`: auto-downloads Moonshine STT model via `download.get_model_for_language(...)`, uses async TTS playback thread.
- `voice_chat_v03.py`: current primary flow from root README quick start; feedback-loop protection via `is_speaking` flag and post-playback cooldown.

## Key codebase conventions

- Keep everything local-first: STT, LLM (Ollama), and TTS all run locally; no cloud inference path is implemented.
- Constrain assistant verbosity through `SYSTEM_PROMPT` and short-generation Ollama options (`num_predict`, `repeat_penalty`, moderate temperature).
- Ignore tiny/noisy transcripts (`len(user_text) < 3`) before calling the LLM.
- Cap chat context size (`chat_history` trimmed to last 10 messages) to keep latency and context stable.
- Always normalize generated audio by peak before playback, then apply a fixed gain scalar.
- Preserve feedback-loop protection when editing runtime flow (`is_speaking`, cooldown timing, and listener guard order).
- Keep script-level configuration constants near the top (`OLLAMA_MODEL`, `VOICE`, model path/arch), matching existing style.

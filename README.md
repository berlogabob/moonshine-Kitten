# moonshine-Kitten Voice Chat

A lightweight, real-time voice chat application built with Python using fully local models:
- **Speech-to-Text (STT)** — Moonshine Voice (real-time microphone transcription) 
- **Language Model (LLM)** — Ollama (local inference) 
- **Text-to-Speech (TTS)** — KittenTTS (ONNX-based voice synthesis)

## Features
- Real-time voice input → text → LLM response → spoken output 
- Feedback loop protection (AI does not hear itself) 
- Fully local processing (no cloud APIs after initial setup) 
- Easy switching of Ollama models and TTS voices


## Requirements
### 1. System dependencies
**macOS (Apple Silicon recommended)**
```bash # Required by phonemizer (used inside KittenTTS) brew install espeak-ng ```
**Windows**
```powershell # Recommended via Chocolatey choco install espeak-ng
# Or download manually: # https://github.com/espeak-ng/espeak-ng/releases ```
**Ubuntu / Debian**
```bash sudo apt update sudo apt install espeak-ng libespeak-ng1 ```

### 2. Python & uv
- Python 3.10 – 3.13 - [uv](https://docs.astral.sh/uv/getting-started/installation/) — fast modern Python project & dependency manager
Install uv if you don't have it yet:
```bash # macOS / Linux curl -LsSf https://astral.sh/uv/install.sh "| sh
# Windows (PowerShell) powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 "| iex" ```

### 3. Ollama (local LLM backend)
- Download and install: https://ollama.com - Start Ollama server: run `ollama serve` or open Ollama.app - Pull at least one model (example):
```bash ollama pull gemma2:2b
# other good small models: # ollama pull phi3:mini # ollama pull llama3.2:3b # ollama pull tinyllama:1.1b ```


## Quick Start
```bash # 1. Clone the repository git clone https://github.com/berlogabob/moonshine-Kitten.git cd moonshine-Kitten
# 2. Install all Python dependencies (creates .venv automatically) uv sync
# 3. Download the TTS model (~75 MB, one-time) uv run hf download KittenML/kitten-tts-mini-0.8 --local-dir ./kitten-tts-mini-0.8
# 4. Run the voice chat uv run voice_chat_v03.py ```


## Configuration (edit in voice_chat_v03.py)
```python # Which Ollama model to use (must be pulled already) OLLAMA_MODEL = "gemma2:2b"

# alternatives: "phi3:mini", "llama3.2:3b", "tinyllama:1.1b"
# TTS voice (from KittenTTS voices.npz) VOICE = "Jasper"









# try "Bella" or "Luna" if too quiet ```
## Troubleshooting
"| Issue














"| Solution








































"| "|------------------------------------"|-------------------------------------------------------------------------------------------"| "| `espeak not installed`





 "| Install espeak-ng (see above) + verify path in code: `EspeakWrapper.set_library(...)`

 "| "| TTS model file not found




 "| Run the `hf download` command from step 3























 "| "| Ollama connection error





"| Make sure `ollama serve` is running + model is pulled

















 "| "| No sound / very quiet output


 "| Change `VOICE` to "Bella" or "Luna", check system volume & microphone gain






"| "| `ModuleNotFoundError`






"| Run `uv sync` again


































 "| "| Audio feedback / robot hears itself"| Script already has protection — increase `time.sleep(0.8)` if needed










"|
## Project structure
``` moonshine-Kitten/ ├── voice_chat_v03.py


 # Recommended / latest working version ├── voice_chat_v02.py


 # Previous version (for reference) ├── pyproject.toml




# Project metadata & dependencies ├── uv.lock







 # Locked exact versions (reproducible builds) ├── .python-version



 # Suggested Python version └── kitten-tts-mini-0.8/

# TTS model files (downloaded separately — not in git) ```
## Notes
- The TTS model is **not** committed to git (too large). Everyone must download it once. - Tested primarily on macOS (Apple Silicon M-series) with Python 3.12 via uv. - Feel free to open an Issue if something doesn't work on your system.
Enjoy experimenting with local voice AI! Questions → Issues tab|

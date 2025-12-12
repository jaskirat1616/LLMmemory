<p align="center">
  <img src="./logo.png" alt="LLMmemory logo" width="220" />
</p>

# LLMmemory

Local playground for experimenting with chat-style memory systems (buffer, summary, vector-style recall, etc.) powered by an MLX-hosted model. Current default: `mlx-community/Qwen3-VL-8B-Instruct-4bit` (MLX VLM build) [HF card](https://huggingface.co/mlx-community/Qwen3-VL-8B-Instruct-4bit).

## Prerequisites
- macOS with Apple Silicon
- Python 3.11+ (recommended)
- Enough disk/RAM for the chosen model (8B class is several GB on disk)

## Quickstart
```bash
cd /Users/jaskiratsingh/Desktop/LLMmemory
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export MODEL_ID="mlx-community/Qwen3-VL-8B-Instruct-4bit"
export MODEL_KIND="vlm"  # "auto", "vlm", or "lm"
# make sure the package is importable
export PYTHONPATH="src:${PYTHONPATH}"
```

Download the model weights ahead of time if you prefer:
```bash
huggingface-cli download "$MODEL_ID"
```

## Running a chat UI
- Streamlit: `streamlit run src/llmmemory/apps/streamlit_app.py`
- Gradio: `python -m llmmemory.apps.gradio_app`

Both UIs let you switch memory backends at runtime.

## Project layout
- `src/llmmemory/` — Python package
  - `memory/` — pluggable memory strategies (buffer, summary, vector-ish)
  - `models/` — MLX model loader + chat helper
  - `chat/` — session orchestration for model + memory
  - `apps/` — Streamlit and Gradio entrypoints
- `tests/` — slots for unit/integration tests
- `configs/` — place to store preset configs or prompt templates

## Notes on the model
- Default `MODEL_ID` targets the 4-bit Qwen3-VL Instruct MLX build. If you switch to a text-only MLX model, set `MODEL_KIND=lm`.
- The loader uses `trust_remote_code=True` because some MLX chat models ship custom tokenizers/processors; only point it at repos you trust.



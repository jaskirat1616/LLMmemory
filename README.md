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

## 📚 Comprehensive Research & Planning Docs

**We've conducted deep research into AI memory systems and have a strategic plan to evolve this into a major developer tool!**

### Essential Reading (Start Here!)

1. **[RESEARCH_SUMMARY.md](./RESEARCH_SUMMARY.md)** - Market analysis, competitor review, developer needs
   - $24M market opportunity (Mem0 just raised)
   - Gap analysis: No TypeScript-native solution
   - Key pain points and solutions

2. **[STRATEGIC_ROADMAP.md](./STRATEGIC_ROADMAP.md)** - 12-month implementation roadmap
   - Phase-by-phase technical plan
   - TypeScript-native architecture
   - Go-to-market strategy

3. **[IMMEDIATE_ACTION_PLAN.md](./IMMEDIATE_ACTION_PLAN.md)** - 30-day startup guide
   - Week-by-week tasks with code examples
   - Publishing checklist
   - Launch strategy

4. **[MEMORY_SYSTEMS_GUIDE.md](./MEMORY_SYSTEMS_GUIDE.md)** - How memory systems work
   - EntityMemory automatic fact extraction
   - When to use each system
   - Real examples

5. **[FIXES_SUMMARY.md](./FIXES_SUMMARY.md)** - Technical improvements log
   - All bugs fixed and verified
   - Memory system testing results

### 🎯 The Opportunity

Based on extensive research:
- **Market**: AI memory tools are exploding (Mem0: $24M, 41K stars, 186M API calls/quarter)
- **Gap**: No comprehensive TypeScript-native solution exists
- **Pain**: Developers struggle with session amnesia, poor memory management
- **Solution**: Build TypeScript-first memory layer with cognitive architecture

### 🚀 Next Evolution: TypeScript Package

We have a complete plan to build `@llmmemory/core` - a TypeScript-native memory layer:
- ✅ Research complete (see docs above)
- ✅ Architecture designed
- ✅ 4-month roadmap ready
- ✅ Code examples prepared
- 🎯 Target: 10K GitHub stars, 100K npm downloads/month in Year 1

**See [IMMEDIATE_ACTION_PLAN.md](./IMMEDIATE_ACTION_PLAN.md) for complete implementation guide.**

## Next steps
- Review research documents to understand the full opportunity
- Consider TypeScript implementation (detailed plan available)
- Add proper summarization using the model itself
- Plug in real embedding backend for vector recall
- Add benchmarks to compare memory strategies


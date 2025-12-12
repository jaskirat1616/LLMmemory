from dataclasses import dataclass
import os


DEFAULT_MODEL_ID = os.getenv(
    "MODEL_ID", "mlx-community/Qwen3-VL-8B-Instruct-4bit"
)
# auto | lm | vlm
DEFAULT_MODEL_KIND = os.getenv("MODEL_KIND", "auto")


@dataclass
class ModelConfig:
    model_id: str = DEFAULT_MODEL_ID
    model_kind: str = DEFAULT_MODEL_KIND
    max_tokens: int = int(os.getenv("MAX_TOKENS", 512))
    temperature: float = float(os.getenv("TEMPERATURE", 0.7))


SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a concise, helpful assistant. Keep replies brief unless asked.",
)


# Memory system registry
MEMORY_SYSTEMS = {
    "chatgpt": ["conversation_buffer", "entity_memory"],
    "claude": ["project_memory", "knowledge_base"],
    "grok": ["context_aware"],
    "mem0": ["mem0_adapter"],
    "custom": ["buffer", "summary", "vector"],
}

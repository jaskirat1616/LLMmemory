from __future__ import annotations

from typing import Any, Iterable, Tuple

from llmmemory.memory.base import ChatMessage
from llmmemory.config import ModelConfig


def _detect_kind(model_id: str, explicit: str | None = None) -> str:
    if explicit and explicit != "auto":
        return explicit
    lower = model_id.lower()
    if "vl" in lower:
        return "vlm"
    return "lm"


def load_model(
    model_id: str,
    model_kind: str = "auto",
    tokenizer_kwargs: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> Tuple[Any, Any, str]:
    """
    Returns (model, tokenizer_or_processor, kind)
    kind ∈ {"lm", "vlm"}
    """
    tokenizer_kwargs = tokenizer_kwargs or {}
    model_kwargs = model_kwargs or {}
    resolved_kind = _detect_kind(model_id, model_kind)

    if resolved_kind == "vlm":
        try:
            from mlx_vlm import load as load_vlm
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "mlx-vlm is required for VLM models. Install with `pip install mlx-vlm`."
            ) from exc

        model, processor = load_vlm(
            model_id,
            tokenizer_config=tokenizer_kwargs,
            model_config=model_kwargs,
            trust_remote_code=True,
        )
        return model, processor, "vlm"

    # default to text-only LM path
    try:
        from mlx_lm import load as load_lm
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "mlx-lm is required for text models. Install with `pip install mlx-lm`."
        ) from exc

    model, tokenizer = load_lm(
        model_id,
        tokenizer_config=tokenizer_kwargs,
        model_config=model_kwargs,
        trust_remote_code=True,
    )
    return model, tokenizer, "lm"


def format_prompt(messages: Iterable[ChatMessage], tokenizer: Any) -> str:
    chat = [{"role": msg.role, "content": msg.content} for msg in messages]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

    # fallback to a simple plain-text chat format
    prompt_parts = []
    for msg in chat:
        prompt_parts.append(f"{msg['role']}: {msg['content']}")
    prompt_parts.append("assistant:")
    return "\n".join(prompt_parts)


def generate_reply(
    model: Any,
    tokenizer: Any,
    messages: Iterable[ChatMessage],
    config: ModelConfig,
    model_kind: str = "auto",
) -> str:
    resolved_kind = _detect_kind(config.model_id, model_kind)
    prompt = format_prompt(messages, tokenizer)

    if resolved_kind == "vlm":
        try:
            from mlx_vlm import generate as generate_vlm
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "mlx-vlm is required for VLM models. Install with `pip install mlx-vlm`."
            ) from exc

        output = generate_vlm(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=config.max_tokens,
            temp=config.temperature,
            stream=False,
        )
    else:
        try:
            from mlx_lm import generate as generate_lm
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "mlx-lm is required for text models. Install with `pip install mlx-lm`."
            ) from exc

        output = generate_lm(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=config.max_tokens,
            temp=config.temperature,
            stream=False,
        )

    return output.strip() if isinstance(output, str) else str(output)


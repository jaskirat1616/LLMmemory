from __future__ import annotations

import gradio as gr

from llmmemory.chat import ChatSession
from llmmemory.config import ModelConfig
from llmmemory.memory import BufferMemory, SummaryMemory, VectorMemory
from llmmemory.models.loader import load_model


MEMORY_FACTORIES = {
    "buffer": lambda: BufferMemory(max_messages=50),
    "summary": lambda: SummaryMemory(buffer_size=8, summarize_every=8),
    "vector": lambda: VectorMemory(dim=512, max_messages=200),
}


def _build_session(memory_kind: str, cached_model, model_kind: str, cfg: ModelConfig):
    model, tokenizer, _ = cached_model
    memory = MEMORY_FACTORIES[memory_kind]()
    return ChatSession(
        memory, model, tokenizer, model_kind=model_kind, model_config=cfg
    )


def main():
    cfg = ModelConfig()
    cached_model = load_model(cfg.model_id, model_kind=cfg.model_kind)
    model_kind = cached_model[2]

    with gr.Blocks() as demo:
        gr.Markdown("# LLMmemory — MLX + Gradio")
        memory_kind = gr.Dropdown(
            choices=list(MEMORY_FACTORIES.keys()),
            value="buffer",
            label="Memory backend",
        )
        state = gr.State(
            {"session": _build_session("buffer", cached_model, model_kind, cfg)}
        )
        chat = gr.Chatbot(type="messages")
        msg = gr.Textbox(label="Your message", placeholder="Ask something...")
        send = gr.Button("Send")
        clear = gr.Button("Clear")

        def respond(user_message, state_data, memory_kind):
            session: ChatSession = state_data["session"]
            # rebuild session if memory backend changed
            if state_data.get("kind") != memory_kind:
                session = _build_session(memory_kind, cached_model, model_kind, cfg)
            session.add_user_message(user_message)
            reply = session.generate_assistant_reply()
            state_data = {"session": session, "kind": memory_kind}
            history = [
                {"role": msg.role, "content": msg.content}
                for msg in session.memory.get_context()
                if msg.role != "system"
            ]
            return history, state_data, ""

        send.click(
            respond,
            inputs=[msg, state, memory_kind],
            outputs=[chat, state, msg],
        )

        clear.click(
            lambda mk: (
                [],
                {
                    "session": _build_session(mk, cached_model, model_kind, cfg),
                    "kind": mk,
                },
                "",
            ),
            inputs=memory_kind,
            outputs=[chat, state, msg],
            queue=False,
        )

    demo.launch()


if __name__ == "__main__":
    main()


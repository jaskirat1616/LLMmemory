from __future__ import annotations

import gradio as gr

from llmmemory.chat import ChatSession
from llmmemory.config import MEMORY_SYSTEMS, ModelConfig
from llmmemory.memory.factory import create_memory
from llmmemory.models.loader import load_model


def _build_session(
    memory_system: str,
    memory_implementation: str,
    cached_model,
    model_kind: str,
    cfg: ModelConfig,
):
    model, tokenizer, _ = cached_model
    memory = create_memory(
        system=memory_system,
        implementation=memory_implementation,
        model=model,
        tokenizer=tokenizer,
        model_config=cfg,
    )
    return ChatSession(memory, model, tokenizer, model_kind=model_kind, model_config=cfg)


def main():
    cfg = ModelConfig()
    cached_model = load_model(cfg.model_id, model_kind=cfg.model_kind)
    model_kind = cached_model[2]

    with gr.Blocks() as demo:
        gr.Markdown("# LLMmemory — Multi-Memory System Playground")
        
        with gr.Row():
            with gr.Column(scale=1):
                memory_system = gr.Dropdown(
                    choices=list(MEMORY_SYSTEMS.keys()),
                    value="custom",
                    label="Memory System",
                )
                memory_implementation = gr.Dropdown(
                    choices=MEMORY_SYSTEMS["custom"],
                    value="buffer",
                    label="Implementation",
                )
                
                # Update implementation choices when system changes
                def update_implementations(system):
                    return gr.Dropdown(
                        choices=MEMORY_SYSTEMS[system],
                        value=MEMORY_SYSTEMS[system][0],
                    )
                
                memory_system.change(
                    update_implementations,
                    inputs=memory_system,
                    outputs=memory_implementation,
                )
        
        state = gr.State(
            {
                "session": _build_session(
                    "custom", "buffer", cached_model, model_kind, cfg
                ),
                "system": "custom",
                "implementation": "buffer",
            }
        )
        chat = gr.Chatbot(type="messages")
        msg = gr.Textbox(label="Your message", placeholder="Ask something...")
        send = gr.Button("Send")
        clear = gr.Button("Clear")

        def respond(user_message, state_data, system, implementation):
            session: ChatSession = state_data["session"]
            # Rebuild session if memory backend changed
            if (
                state_data.get("system") != system
                or state_data.get("implementation") != implementation
            ):
                session = _build_session(system, implementation, cached_model, model_kind, cfg)
            session.add_user_message(user_message)
            reply = session.generate_assistant_reply()
            state_data = {
                "session": session,
                "system": system,
                "implementation": implementation,
            }
            history = [
                {"role": msg.role, "content": msg.content}
                for msg in session.memory.get_context()
                if msg.role != "system"
            ]
            return history, state_data, ""

        send.click(
            respond,
            inputs=[msg, state, memory_system, memory_implementation],
            outputs=[chat, state, msg],
        )

        def clear_chat(system, implementation):
            return (
                [],
                {
                    "session": _build_session(
                        system, implementation, cached_model, model_kind, cfg
                    ),
                    "system": system,
                    "implementation": implementation,
                },
                "",
            )

        clear.click(
            clear_chat,
            inputs=[memory_system, memory_implementation],
            outputs=[chat, state, msg],
            queue=False,
        )

    demo.launch()


if __name__ == "__main__":
    main()

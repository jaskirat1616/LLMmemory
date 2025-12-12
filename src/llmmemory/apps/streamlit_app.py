import streamlit as st

from llmmemory.chat import ChatSession
from llmmemory.config import MEMORY_SYSTEMS, ModelConfig
from llmmemory.memory.factory import create_memory
from llmmemory.models.loader import load_model


@st.cache_resource(show_spinner=True)
def _load_model_cached(model_id: str, model_kind: str):
    return load_model(model_id, model_kind=model_kind)


def _ensure_session(memory_system: str, memory_implementation: str) -> ChatSession:
    cfg = st.session_state.get("model_config") or ModelConfig()
    st.session_state.model_config = cfg

    model, tokenizer, kind = _load_model_cached(cfg.model_id, cfg.model_kind)

    # Create unique key for session
    session_key = f"{memory_system}:{memory_implementation}:{kind}:{cfg.model_id}"

    if (
        "chat_session" not in st.session_state
        or st.session_state.get("session_key") != session_key
    ):
        # Create memory using factory
        memory = create_memory(
            system=memory_system,
            implementation=memory_implementation,
            model=model,
            tokenizer=tokenizer,
            model_config=cfg,
        )
        st.session_state.session_key = session_key
        st.session_state.chat_session = ChatSession(
            memory=memory,
            model=model,
            tokenizer=tokenizer,
            model_kind=kind,
            model_config=cfg,
        )
    return st.session_state.chat_session


def main():
    st.set_page_config(page_title="LLMmemory (MLX)", page_icon="🧠", layout="wide")
    st.title("LLMmemory — Multi-Memory System Playground")

    with st.sidebar:
        st.markdown("### Memory System")

        # First dropdown: Select memory system
        memory_system = st.selectbox(
            "Memory System",
            options=list(MEMORY_SYSTEMS.keys()),
            format_func=lambda x: x.capitalize(),
            index=4,  # Default to "custom"
        )

        # Second dropdown: Select implementation within system
        implementations = MEMORY_SYSTEMS[memory_system]
        memory_implementation = st.selectbox(
            "Implementation",
            options=implementations,
            format_func=lambda x: x.replace("_", " ").title(),
        )

        # Display README if available
        try:
            import os
            readme_path = f"src/llmmemory/memory/{memory_system}/README.md"
            if os.path.exists(readme_path):
                with open(readme_path, "r") as f:
                    readme_content = f.read()
                with st.expander("📖 About this memory system"):
                    st.markdown(readme_content)
        except Exception:
            pass

        st.markdown("---")
        st.markdown("### Model Settings")
        cfg = st.session_state.get("model_config") or ModelConfig()
        cfg.temperature = st.slider("Temperature", 0.0, 1.5, cfg.temperature, 0.05)
        cfg.max_tokens = st.number_input("Max tokens", 16, 1024, cfg.max_tokens, 8)
        st.session_state.model_config = cfg

        # Show memory stats if available
        st.markdown("---")
        st.markdown("### Memory Stats")
        if "chat_session" in st.session_state:
            memory = st.session_state.chat_session.memory
            context = memory.get_context()
            st.metric("Messages in Context", len(context))

            # Show facts if EntityMemory
            if hasattr(memory, "get_facts"):
                facts = memory.get_facts()
                st.metric("Facts Stored", len(facts))
                if facts and st.checkbox("Show stored facts"):
                    for fact in facts[-5:]:  # Show last 5 facts
                        st.caption(f"• {fact.fact}")

    session = _ensure_session(memory_system, memory_implementation)

    # Display history
    for msg in session.memory.get_context():
        if msg.role == "system":
            continue
        with st.chat_message(msg.role):
            st.markdown(msg.content)

    user_input = st.chat_input("Message the model...")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        session.add_user_message(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Generating..."):
                reply = session.generate_assistant_reply()
                st.markdown(reply)


if __name__ == "__main__":
    main()

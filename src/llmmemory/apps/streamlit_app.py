import streamlit as st

from llmmemory.chat import ChatSession
from llmmemory.config import ModelConfig
from llmmemory.memory import BufferMemory, SummaryMemory, VectorMemory
from llmmemory.models.loader import load_model


MEMORY_FACTORIES = {
    "buffer": lambda: BufferMemory(max_messages=50),
    "summary": lambda: SummaryMemory(buffer_size=8, summarize_every=8),
    "vector": lambda: VectorMemory(dim=512, max_messages=200),
}


@st.cache_resource(show_spinner=True)
def _load_model_cached(model_id: str, model_kind: str):
    return load_model(model_id, model_kind=model_kind)


def _ensure_session(memory_kind: str) -> ChatSession:
    cfg = st.session_state.get("model_config") or ModelConfig()
    st.session_state.model_config = cfg

    model, tokenizer, kind = _load_model_cached(cfg.model_id, cfg.model_kind)

    if (
        "chat_session" not in st.session_state
        or st.session_state.get("memory_kind") != memory_kind
        or st.session_state.get("model_kind") != kind
        or st.session_state.get("model_id") != cfg.model_id
    ):
        memory = MEMORY_FACTORIES[memory_kind]()
        st.session_state.memory_kind = memory_kind
        st.session_state.model_kind = kind
        st.session_state.model_id = cfg.model_id
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
    st.title("LLMmemory — MLX local chat")

    with st.sidebar:
        st.markdown("### Settings")
        memory_kind = st.radio(
            "Memory backend",
            options=list(MEMORY_FACTORIES.keys()),
            format_func=lambda x: x.capitalize(),
            index=0,
        )
        cfg = st.session_state.get("model_config") or ModelConfig()
        cfg.temperature = st.slider("Temperature", 0.0, 1.5, cfg.temperature, 0.05)
        cfg.max_tokens = st.number_input("Max tokens", 16, 1024, cfg.max_tokens, 8)
        st.session_state.model_config = cfg

    session = _ensure_session(memory_kind)

    # display history
    for msg in session.memory.get_context():
        if msg.role == "system":
            continue
        with st.chat_message(msg.role):
            st.write(msg.content)

    user_input = st.chat_input("Message the model...")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        session.add_user_message(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Generating..."):
                reply = session.generate_assistant_reply()
                st.write(reply)


if __name__ == "__main__":
    main()


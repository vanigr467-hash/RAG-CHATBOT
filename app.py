"""
app.py

Streamlit entry point for the RAG Chatbot application. Wires together the
sidebar (PDF upload, RAG/LLM settings, knowledge base controls) and the
main chat interface, delegating all heavy lifting to the backend modules.
"""

from __future__ import annotations

import streamlit as st

from backend.document_processor import PDFProcessingError, process_uploaded_pdfs
from backend.embeddings import DEFAULT_EMBEDDING_MODEL, get_embedding_model
from backend.llm import LLMConfigError, get_configured_model_name, get_groq_api_key, get_llm
from backend.rag_pipeline import RAGPipelineError, answer_question
from backend.vector_store import (
    VectorStoreError,
    create_vector_store,
    load_vector_store,
    reset_vector_store_directory,
    save_vector_store,
)

FAISS_INDEX_PATH = "faiss_index"

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")


# --------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------
def init_session_state() -> None:
    defaults = {
        "messages": [],           # chat history: list of {role, content, sources, knowledge_type}
        "vector_store": None,     # in-memory FAISS index
        "kb_ready": False,        # knowledge base status flag
        "kb_num_documents": 0,
        "kb_num_chunks": 0,
        "confirm_reset": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


def restore_previous_index_if_available() -> None:
    """
    On first load, try to restore a FAISS index previously saved to disk
    in this environment (e.g. from an earlier run in the same session/
    container). Silently does nothing if no saved index exists or it
    fails to load.
    """
    if st.session_state.kb_ready or st.session_state.vector_store is not None:
        return
    try:
        embedding_model = get_embedding_model(DEFAULT_EMBEDDING_MODEL)
        restored_store = load_vector_store(FAISS_INDEX_PATH, embedding_model)
        st.session_state.vector_store = restored_store
        st.session_state.kb_ready = True
        st.session_state.kb_num_chunks = restored_store.index.ntotal
    except VectorStoreError:
        pass


restore_previous_index_if_available()


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Only PDF files are accepted.",
    )

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) selected:")
        for f in uploaded_files:
            size_kb = len(f.getvalue()) / 1024
            st.caption(f"• {f.name} ({size_kb:.1f} KB)")

    MAX_FILE_SIZE_MB = 25
    oversized = [
        f.name for f in (uploaded_files or []) if len(f.getvalue()) > MAX_FILE_SIZE_MB * 1024 * 1024
    ]
    if oversized:
        st.error(
            f"These files exceed the {MAX_FILE_SIZE_MB}MB limit and will be skipped: "
            f"{', '.join(oversized)}"
        )

    st.divider()
    st.header("⚙️ RAG Settings")
    chunk_size = st.slider("Chunk Size", min_value=200, max_value=3000, value=1000, step=50)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=150, step=10)
    top_k = st.slider("Number of retrieved chunks (k)", min_value=1, max_value=10, value=4)
    st.caption("Similarity search: FAISS")

    st.divider()
    st.header("🤖 LLM Settings")
    configured_model = get_configured_model_name()
    st.text_input("Model name", value=configured_model, disabled=True)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    max_tokens = st.slider("Max response tokens", min_value=128, max_value=4096, value=1024, step=128)

    if not get_groq_api_key():
        st.warning("GROQ_API_KEY is not set. Add it to your .env file to enable answers.")

    st.divider()
    st.header("📚 Knowledge Base")
    if st.session_state.kb_ready:
        st.success("Status: 🟢 Ready")
        st.caption(f"Documents: {st.session_state.kb_num_documents}")
        st.caption(f"Chunks: {st.session_state.kb_num_chunks}")
    else:
        st.info("Status: 🔴 Not Ready")

    build_clicked = st.button("🔨 Build Knowledge Base", use_container_width=True)
    clear_chat_clicked = st.button("🗑️ Clear Chat", use_container_width=True)
    reset_kb_clicked = st.button("♻️ Reset Knowledge Base", use_container_width=True)


# --------------------------------------------------------------------------
# Sidebar button actions
# --------------------------------------------------------------------------
if build_clicked:
    valid_files = [
        f for f in (uploaded_files or [])
        if len(f.getvalue()) <= MAX_FILE_SIZE_MB * 1024 * 1024
    ]
    if not valid_files:
        st.sidebar.error("Please upload at least one valid PDF before building the knowledge base.")
    else:
        try:
            with st.spinner("Extracting and chunking documents..."):
                chunks = process_uploaded_pdfs(
                    valid_files, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )

            with st.spinner("Generating embeddings and building FAISS index..."):
                embedding_model = get_embedding_model(DEFAULT_EMBEDDING_MODEL)
                vector_store = create_vector_store(chunks, embedding_model)
                save_vector_store(vector_store, FAISS_INDEX_PATH)

            st.session_state.vector_store = vector_store
            st.session_state.kb_ready = True
            st.session_state.kb_num_documents = len(valid_files)
            st.session_state.kb_num_chunks = len(chunks)
            st.sidebar.success(f"Knowledge base built: {len(valid_files)} document(s), {len(chunks)} chunks.")
        except PDFProcessingError as exc:
            st.sidebar.error(str(exc))
        except VectorStoreError as exc:
            st.sidebar.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error("An unexpected error occurred while building the knowledge base.")

if clear_chat_clicked:
    st.session_state.messages = []
    st.rerun()

if reset_kb_clicked:
    st.session_state.confirm_reset = True

if st.session_state.confirm_reset:
    st.sidebar.warning("This will permanently delete the current knowledge base.")
    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        if st.button("✅ Confirm Reset", use_container_width=True):
            reset_vector_store_directory(FAISS_INDEX_PATH)
            st.session_state.vector_store = None
            st.session_state.kb_ready = False
            st.session_state.kb_num_documents = 0
            st.session_state.kb_num_chunks = 0
            st.session_state.confirm_reset = False
            st.sidebar.success("Knowledge base reset.")
            st.rerun()
    with col_b:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.confirm_reset = False
            st.rerun()


# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
st.title("🤖 RAG Chatbot")
st.caption("Upload your PDFs and ask questions using document knowledge + LLM intelligence.")
st.subheader("Ask questions about your documents")

if not st.session_state.kb_ready:
    st.info(
        "No knowledge base yet. Upload PDFs in the sidebar and click "
        "**🔨 Build Knowledge Base** to get started, or just start chatting "
        "to use general LLM knowledge."
    )

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            knowledge_type = message.get("knowledge_type")
            if knowledge_type == "document":
                st.caption("📚 Document Context")
            elif knowledge_type == "document + general":
                st.caption("📚 Document Context + 🌐 General Knowledge")
            elif knowledge_type == "general":
                st.caption("🌐 General LLM Knowledge")

            sources = message.get("sources") or []
            if sources:
                with st.expander(f"📚 Sources ({len(sources)})"):
                    for i, src in enumerate(sources, start=1):
                        st.markdown(f"**{i}. {src['source']} — Page {src['page']}**")
                        preview = src["content"]
                        if len(preview) > 400:
                            preview = preview[:400] + "..."
                        st.markdown(f"> {preview}")

# Chat input
user_question = st.chat_input("Ask a question about your documents...")

if user_question:
    if not user_question.strip():
        st.warning("Please enter a non-empty question.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            api_key = get_groq_api_key()
            if not api_key:
                error_msg = "Groq API key is not configured. Please add GROQ_API_KEY to your .env file."
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg, "sources": [], "knowledge_type": None}
                )
            else:
                with st.spinner("Thinking..."):
                    try:
                        llm = get_llm(
                            model_name=configured_model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        result = answer_question(
                            question=user_question,
                            vector_store=st.session_state.vector_store,
                            llm=llm,
                            k=top_k,
                            chat_history=st.session_state.messages[:-1],
                        )

                        st.markdown(result["answer"])

                        knowledge_type = result["knowledge_type"]
                        if knowledge_type == "document":
                            st.caption("📚 Document Context")
                        elif knowledge_type == "document + general":
                            st.caption("📚 Document Context + 🌐 General Knowledge")
                        else:
                            st.caption("🌐 General LLM Knowledge")

                        if result["sources"]:
                            with st.expander(f"📚 Sources ({len(result['sources'])})"):
                                for i, src in enumerate(result["sources"], start=1):
                                    st.markdown(f"**{i}. {src['source']} — Page {src['page']}**")
                                    preview = src["content"]
                                    if len(preview) > 400:
                                        preview = preview[:400] + "..."
                                    st.markdown(f"> {preview}")
                        elif st.session_state.kb_ready:
                            st.caption(
                                "I couldn't find relevant information about this topic in the "
                                "uploaded documents, so I answered using general LLM knowledge."
                            )

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": result["answer"],
                                "sources": result["sources"],
                                "knowledge_type": knowledge_type,
                            }
                        )
                    except LLMConfigError as exc:
                        st.error(str(exc))
                        st.session_state.messages.append(
                            {"role": "assistant", "content": str(exc), "sources": [], "knowledge_type": None}
                        )
                    except RAGPipelineError as exc:
                        st.error(str(exc))
                        st.session_state.messages.append(
                            {"role": "assistant", "content": str(exc), "sources": [], "knowledge_type": None}
                        )
                    except Exception as exc:  # noqa: BLE001
                        friendly = "An unexpected error occurred while generating the answer. Please try again."
                        st.error(friendly)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": friendly, "sources": [], "knowledge_type": None}
                        )


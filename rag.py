import os
import tempfile
import streamlit as st
import pandas as pd
import requests

from streamlit_lottie import st_lottie
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader
)
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------------------------------------------------------
# Page setup & Custom CSS Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Document Intelligence Hub",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for gorgeous modern styling, cards, gradients, and fonts
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Main Background & Sidebar styling */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #090d16;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Gradient Header Title */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }

    /* Custom Glassmorphism Containers / Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }

    /* Style file uploader box */
    [data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.6);
        border: 2px dashed rgba(129, 140, 248, 0.4);
        border-radius: 12px;
        padding: 15px;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #818cf8;
        background: rgba(30, 41, 59, 0.8);
    }

    /* Customize Chat Bubbles */
    .stChatMessage {
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to load Lottie animations securely
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# -----------------------------------------------------------------------------
# Sidebar: API key + Advanced Settings
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Control Center")
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    
    st.markdown("---")
    st.markdown("#### 🧠 Model Parameters")
    model_name = st.selectbox("Chat Model", ["gpt-4o-mini", "gpt-4o"])
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 400, 150, step=50)
    top_k = st.slider("Retrieved Chunks (k)", 2, 10, 4)
    
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 12px;'>Crafted with ❤️ using Streamlit & LangChain</p>", unsafe_allow_html=True)

if not api_key:
    # Header & Hero Layout when not logged in
    st.markdown('<p class="main-title">✨ AI Document Intelligence Hub</p>', unsafe_allow_html=True)
    st.caption("Chat seamlessly with your PDF, Word DOCX, and Excel spreadsheets using advanced RAG technology.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
            <div class="glass-card" style="margin-top: 20px;">
                <h3>👋 Welcome!</h3>
                <p>Unlock the insights locked inside your documents instantly. This app extracts data, splits text with precision, and answers questions with precise context citations.</p>
                <ul>
                    <li><b>Supported Formats:</b> PDF, DOCX, XLSX, XLS</li>
                    <li><b>Security:</b> Private vector search powered by FAISS</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        st.warning("👈 Please enter your **OpenAI API Key** in the sidebar to activate the system.")
    with col2:
        # Load animated lottie robot/AI icon
        lottie_ai = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_jcikwtux.json")
        if lottie_ai:
            st_lottie(lottie_ai, height=220, key="ai_robot")
            
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# -----------------------------------------------------------------------------
# Main Application Interface Header
# -----------------------------------------------------------------------------
st.markdown('<p class="main-title">✨ AI Document Intelligence Hub</p>', unsafe_allow_html=True)
st.caption("Upload files below and query your knowledge base interactively.")

# File uploader card
uploaded_file = st.file_uploader("📂 Drop your file here", type=["pdf", "docx", "doc", "xlsx", "xls"])

def build_vectorstore(file_bytes: bytes, file_extension: str, size: int, overlap: int) -> FAISS:
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        if file_extension == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif file_extension in [".docx", ".doc"]:
            loader = Docx2txtLoader(tmp_path)
        elif file_extension in [".xlsx", ".xls"]:
            loader = UnstructuredExcelLoader(tmp_path, mode="elements")
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
            
        docs = loader.load()
    finally:
        os.path.isfile(tmp_path) and os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(chunks, embeddings)

if uploaded_file is not None:
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    file_sig = (uploaded_file.name, uploaded_file.size, chunk_size, chunk_overlap)
    
    if st.session_state.get("file_sig") != file_sig:
        with st.spinner(f"🚀 Indexing and vectorizing {uploaded_file.name}..."):
            st.session_state.vectorstore = build_vectorstore(
                uploaded_file.getvalue(), file_ext, chunk_size, chunk_overlap
            )
            st.session_state.file_sig = file_sig
            st.session_state.messages = []
        st.success(f"🎉 Successfully indexed **{uploaded_file.name}**! Ready for questions.")

# -----------------------------------------------------------------------------
# RAG Chain Configuration
# -----------------------------------------------------------------------------
RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert AI document analyst. Answer questions strictly using the "
        "provided context from the uploaded document.\n"
        "Rules:\n"
        "1. Answer ONLY using the provided text context.\n"
        '2. If the answer is not present, respond with: "I couldn\'t find that information in the document."\n'
        "3. Maintain a professional tone and cite context chunks when helpful.\n\n"
        "Context:\n{context}",
    ),
    ("human", "{question}"),
])

def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)

def get_chain(vectorstore: FAISS, model: str, k: int):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model=model, temperature=0)
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    ), retriever

# -----------------------------------------------------------------------------
# Chat Interface Stream
# -----------------------------------------------------------------------------
if "vectorstore" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("💬 Ask a question about your uploaded document...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        chain, retriever = get_chain(
            st.session_state.vectorstore, model_name, top_k
        )

        with st.chat_message("assistant"):
            with st.spinner("🤖 Analyzing document chunks..."):
                answer = chain.invoke(question)
                st.markdown(answer)

            with st.expander("🔍 View Retrieved Sources"):
                for doc in retriever.invoke(question):
                    st.text(doc.page_content[:400] + "...")
                    st.divider()

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("👈 Please upload a document using the uploader above to unlock the chat interface.")
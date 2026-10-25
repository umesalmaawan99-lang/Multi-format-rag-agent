# 📄 Multi-Format RAG Agent (PDF, DOCX, Excel)

A modern, full-stack **Retrieval-Augmented Generation (RAG)** web application built with **Streamlit**, **LangChain**, **FAISS**, and **OpenAI**. This app allows users to upload various document formats (**PDF**, **Word DOCX**, and **Excel spreadsheets**) and chat with them contextually with source citations and a sleek glassmorphism UI.

---

## 🚀 Features

* **Multi-Format Support:** Upload and query PDFs, Word documents (`.docx`, `.doc`), and Excel files (`.xlsx`, `.xls`).
* **Advanced Chunking & Embedding:** Powered by LangChain's `RecursiveCharacterTextSplitter` and OpenAI's `text-embedding-3-small`.
* **Private Vector Search:** Fast and efficient local similarity search using **FAISS**.
* **Modern UI/UX:** Styled with custom CSS glassmorphism, animated Lottie headers, and interactive dark-mode components.
* **Source Transparency:** Expandable accordion showing exact retrieved chunks and page/source citations for every answer.

---

## 🛠️ Tech Stack

* **Frontend/UI:** Streamlit, Streamlit-Lottie
* **Orchestration:** LangChain (LCEL - LangChain Expression Language)
* **Vector Store:** FAISS (Facebook AI Similarity Search)
* **LLM & Embeddings:** OpenAI (`gpt-4o-mini`, `gpt-4o`, `text-embedding-3-small`)
* **Document Parsers:** PyPDF, `python-docx`, `openpyxl`, Unstructured

---

## ⚙️ Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/multi-format-rag-agent.git](https://github.com/YOUR_USERNAME/multi-format-rag-agent.git)
   cd multi-format-rag-agent

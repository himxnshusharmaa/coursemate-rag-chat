import hashlib
import io
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

st.set_page_config(
    page_title="CourseMate RAG Book Chat",
    page_icon="📚",
    layout="wide",
)

DB_DIRECTORY = Path("chroma_db")


@st.cache_resource
def get_embedding_model():
    """Load the embedding model once, then reuse it."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={
            "batch_size": 32,
            "normalize_embeddings": True,
        },
    )


def extract_pdf_documents(file_bytes: bytes, filename: str) -> list[Document]:
    """Read text from each page of an uploaded PDF."""
    reader = PdfReader(io.BytesIO(file_bytes))
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "page": page_number,
                    },
                )
            )

    return documents


def build_vectorstore(file_bytes: bytes, filename: str):
    """Create or reuse a Chroma collection for the uploaded book."""
    documents = extract_pdf_documents(file_bytes, filename)

    if not documents:
        raise ValueError(
            "No readable text was found. This may be a scanned PDF image."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(documents)

    book_hash = hashlib.md5(file_bytes).hexdigest()
    collection_name = f"book_{book_hash}"

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=get_embedding_model(),
        persist_directory=str(DB_DIRECTORY),
    )

    # Reuse the database when this exact book was already processed.
    existing_chunks = vectorstore._collection.count()
    if existing_chunks > 0:
        return vectorstore, existing_chunks, True

    progress = st.progress(0, text="Creating embeddings...")
    batch_size = 32

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectorstore.add_documents(batch)

        completed = min((start + batch_size) / len(chunks), 1.0)
        progress.progress(
            completed,
            text=f"Creating embeddings: {min(start + batch_size, len(chunks))}/{len(chunks)} chunks",
        )

    progress.empty()

    return vectorstore, len(chunks), False


def answer_question(question: str, vectorstore, top_k: int, temperature: float):
    """Retrieve relevant chunks and generate a Groq answer."""
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": top_k,
            "fetch_k": max(top_k * 3, 10),
            "lambda_mult": 0.5,
        },
    )

    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('source', 'Unknown')} | "
            f"Page: {doc.metadata.get('page', 'Unknown')}\n"
            f"{doc.page_content}"
            for doc in retrieved_docs
        ]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant answering questions about an uploaded book.

Use only the provided context.
If the answer is not present in the context, say exactly:
"I could not find the answer in the uploaded book."

Give a clear, helpful answer and mention page numbers when possible.""",
            ),
            (
                "human",
                """Context:
{context}

Question:
{question}""",
            ),
        ]
    )

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=temperature,
    )

    response = (prompt | llm).invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return response.content, retrieved_docs


# ---------------- UI ---------------- #

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "book_name" not in st.session_state:
    st.session_state.book_name = None


st.title("📚 RAG Book Chat")
st.caption("Upload a PDF book, then ask questions from its contents.")


with st.sidebar:
    st.header("Book settings")

    uploaded_file = st.file_uploader(
        "Upload a PDF book",
        type=["pdf"],
    )

    top_k = st.slider(
        "Number of relevant chunks",
        min_value=2,
        max_value=8,
        value=4,
    )

    temperature = st.slider(
        "Answer creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
    )

    if st.button("Process Book", use_container_width=True):
        if uploaded_file is None:
            st.warning("Please upload a PDF first.")
        else:
            try:
                with st.spinner("Reading your book..."):
                    vectorstore, chunk_count, reused = build_vectorstore(
                        uploaded_file.getvalue(),
                        uploaded_file.name,
                    )

                st.session_state.vectorstore = vectorstore
                st.session_state.book_name = uploaded_file.name
                st.session_state.messages = []

                if reused:
                    st.success(
                        f"Book loaded from existing database ({chunk_count} chunks)."
                    )
                else:
                    st.success(
                        f"Book processed successfully! Created {chunk_count} chunks."
                    )

            except Exception as error:
                st.error(f"Could not process this PDF: {error}")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Model: openai/gpt-oss-20b")


if st.session_state.book_name:
    st.info(f"Currently chatting with: **{st.session_state.book_name}**")
else:
    st.info("Upload a PDF from the sidebar to begin.")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input("Ask a question about the uploaded book...")

if question:
    if st.session_state.vectorstore is None:
        st.warning("Please upload and process a PDF book first.")

    elif not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is missing. Add it to your .env file.")

    else:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the book and writing an answer..."):
                try:
                    answer, source_docs = answer_question(
                        question=question,
                        vectorstore=st.session_state.vectorstore,
                        top_k=top_k,
                        temperature=temperature,
                    )

                    st.markdown(answer)

                    with st.expander("View retrieved pages"):
                        for index, document in enumerate(source_docs, start=1):
                            page_number = document.metadata.get("page", "Unknown")
                            st.markdown(f"**Result {index} — Page {page_number}**")
                            st.write(document.page_content[:700] + "...")

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

                except Exception as error:
                    st.error(f"Something went wrong: {error}")
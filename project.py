import json
import requests
import streamlit as st
from pathlib import Path
from pypdf import PdfReader
from docx import Document
import chromadb


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="NyayaAI",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ NyayaAI")
st.subheader("AI-Powered Legal Assistance and Document Analysis System")

st.info(
    "NyayaAI is an educational legal-document analysis system "
    "and does not provide professional legal advice."
)


# ==========================================
# PROJECT DIRECTORIES
# ==========================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"

DATA_DIR.mkdir(exist_ok=True)


# ==========================================
# OLLAMA CONFIGURATION
# ==========================================

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"


# ==========================================
# OLLAMA FUNCTIONS
# ==========================================

def check_ollama():
    """Check whether Ollama is running."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return response.ok
    except requests.RequestException:
        return False


def create_embedding(text):
    """Create an embedding using Ollama."""
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBEDDING_MODEL,
            "input": text
        },
        timeout=120
    )
    response.raise_for_status()

    data = response.json()
    embeddings = data.get("embeddings")

    if not embeddings:
        raise RuntimeError("Ollama did not return an embedding.")

    return embeddings[0]


def ask_ollama(prompt):
    """Generate an answer using the local Ollama LLM."""
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0
            }
        },
        timeout=180
    )
    response.raise_for_status()

    data = response.json()
    return data.get("response", "").strip()


# ==========================================
# DOCUMENT LOADING
# ==========================================

def load_document(file_path):
    """Extract text from PDF or DOCX."""
    extension = file_path.suffix.lower()
    documents = []

    if extension == ".pdf":
        reader = PdfReader(str(file_path))

        for page_number, page in enumerate(reader.pages):
            text = page.extract_text() or ""

            if text.strip():
                documents.append({
                    "text": text,
                    "source": file_path.name,
                    "page": page_number + 1
                })

    elif extension == ".docx":
        document = Document(str(file_path))

        paragraphs = [
            paragraph.text.strip()
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        text = "\n".join(paragraphs)

        if text.strip():
            documents.append({
                "text": text,
                "source": file_path.name,
                "page": None
            })

    return documents


# ==========================================
# TEXT CHUNKING
# ==========================================

def split_text(text, chunk_size=1000, chunk_overlap=150):
    """Split text into overlapping chunks without LangChain."""
    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(end - chunk_overlap, start + 1)

    return chunks


# ==========================================
# CHROMA DATABASE
# ==========================================

@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=str(DB_DIR))


def get_collection(reset=False):
    client = get_chroma_client()

    if reset:
        try:
            client.delete_collection("nyayaai_documents")
        except Exception:
            pass

    return client.get_or_create_collection(
        name="nyayaai_documents"
    )


# ==========================================
# CREATE VECTOR DATABASE
# ==========================================

def create_vector_database(documents):
    collection = get_collection(reset=True)

    all_chunks = []
    embeddings = []
    ids = []
    metadatas = []

    chunk_number = 0

    for document in documents:
        chunks = split_text(document["text"])

        for chunk in chunks:
            embedding = create_embedding(chunk)

            all_chunks.append(chunk)
            embeddings.append(embedding)
            ids.append(f"chunk_{chunk_number}")

            metadata = {
                "source": document["source"]
            }

            if document["page"] is not None:
                metadata["page"] = document["page"]

            metadatas.append(metadata)
            chunk_number += 1

    if not all_chunks:
        return collection, 0

    collection.add(
        ids=ids,
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return collection, len(all_chunks)


# ==========================================
# RETRIEVE RELEVANT DOCUMENTS
# ==========================================

def search_documents(question, number_of_results=4):
    collection = get_collection()

    if collection.count() == 0:
        return []

    question_embedding = create_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(number_of_results, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    documents = []

    result_documents = results.get("documents", [[]])[0]
    result_metadatas = results.get("metadatas", [[]])[0]
    result_distances = results.get("distances", [[]])[0]

    for text, metadata, distance in zip(
        result_documents,
        result_metadatas,
        result_distances
    ):
        documents.append({
            "text": text,
            "metadata": metadata or {},
            "distance": distance
        })

    return documents


# ==========================================
# PROCESS UPLOADED DOCUMENTS
# ==========================================

uploaded_files = st.file_uploader(
    "Upload Legal Documents",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

if uploaded_files:

    if st.button("Process Documents"):

        if not check_ollama():
            st.error(
                "Ollama is not running. Start Ollama and try again."
            )
        else:
            all_documents = []

            with st.spinner("Reading documents and creating embeddings..."):

                for uploaded_file in uploaded_files:

                    file_path = DATA_DIR / uploaded_file.name

                    with open(file_path, "wb") as file:
                        file.write(uploaded_file.getbuffer())

                    documents = load_document(file_path)
                    all_documents.extend(documents)

                if all_documents:

                    try:
                        database, number_of_chunks = (
                            create_vector_database(all_documents)
                        )

                        st.success(
                            "Documents processed successfully!"
                        )

                        st.write(
                            f"Created {number_of_chunks} text chunks."
                        )

                        st.session_state["documents_ready"] = True

                    except Exception as error:
                        st.error(
                            f"Error while creating the vector database: {error}"
                        )

                else:
                    st.error("Could not read the documents.")


# ==========================================
# CHECK DATABASE
# ==========================================

try:
    collection = get_collection()

    if collection.count() > 0:
        st.session_state["documents_ready"] = True

except Exception:
    pass


st.divider()


# ==========================================
# QUESTION
# ==========================================

question = st.text_input(
    "Ask a question about your legal document",
    placeholder=(
        "Example: What are the main obligations "
        "mentioned in this agreement?"
    )
)


# ==========================================
# QUESTION ANSWERING
# ==========================================

if question:

    if not st.session_state.get("documents_ready", False):

        st.warning(
            "Please upload and process a document first."
        )

    else:

        try:

            if not check_ollama():
                st.error(
                    "Ollama is not running. Please start Ollama."
                )
                st.stop()

            # Retrieve relevant chunks
            relevant_documents = search_documents(
                question,
                number_of_results=4
            )

            if not relevant_documents:

                st.warning(
                    "No relevant information found."
                )

            else:

                # Create context
                context_parts = []

                for document in relevant_documents:

                    source = document["metadata"].get(
                        "source",
                        "Unknown"
                    )

                    page = document["metadata"].get(
                        "page"
                    )

                    if page:
                        source_text = (
                            f"Source: {source}, Page: {page}"
                        )
                    else:
                        source_text = f"Source: {source}"

                    context_parts.append(
                        f"{source_text}\n{document['text']}"
                    )

                context = "\n\n".join(context_parts)

                # ==================================
                # RAG PROMPT
                # ==================================

                prompt = f"""
You are NyayaAI, an educational legal document
analysis assistant.

Use ONLY the information provided in the document context.

Do not invent:
- Laws
- Legal sections
- Facts
- Cases
- Citations
- Legal conclusions

If the answer is not available in the uploaded document,
say:

"The information is not available in the uploaded document."

Explain complicated legal language in simple terms.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

                # Ask local Ollama model
                with st.spinner("Generating answer..."):
                    answer = ask_ollama(prompt)

                # ==================================
                # DISPLAY ANSWER
                # ==================================

                st.markdown("### ⚖️ NyayaAI Answer")
                st.write(answer)

                # ==================================
                # SOURCES
                # ==================================

                st.markdown("### 📄 Retrieved Sources")

                for index, document in enumerate(
                    relevant_documents,
                    start=1
                ):

                    source = Path(
                        document["metadata"].get(
                            "source",
                            "Unknown"
                        )
                    ).name

                    page = document["metadata"].get("page")

                    if page:
                        st.write(
                            f"{index}. {source} (Page {page})"
                        )
                    else:
                        st.write(
                            f"{index}. {source}"
                        )

        except requests.RequestException as error:

            st.error(
                f"Could not connect to Ollama: {error}"
            )

        except Exception as error:

            st.error(
                f"Error: {error}"
            )


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("⚖️ NyayaAI")

    st.write(
        """
NyayaAI uses Retrieval-Augmented
Generation (RAG) to analyze
legal documents.
"""
    )

    st.markdown("### RAG Pipeline")

    st.write("1. Upload PDF/DOCX")
    st.write("2. Extract text")
    st.write("3. Split text into chunks")
    st.write("4. Ollama Embeddings")
    st.write("5. Chroma Vector Database")
    st.write("6. Similarity Search")
    st.write("7. Ollama LLM")
    st.write("8. Generate Answer")

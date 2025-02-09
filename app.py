import os
import time
import asyncio
import streamlit as st

# =============================================================================
# IMPORTS: LangChain & Ollama Libraries for PDF Loading, Text Splitting,
# Vector Storage, Prompt Templating, and Local Model Inference
# =============================================================================
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# =============================================================================
# GLOBAL CONFIGURATION & CONSTANTS
# =============================================================================
# Set up the Streamlit page configuration
st.set_page_config(page_title="PDF-Chat-Wizard", layout="wide")

# Custom CSS for professional styling
st.markdown("""
    <style>
    .stApp { background-color: #121212; color: #e0e0e0; }
    h1, h2, h3 { color: #00FFAA; }
    .stButton>button { background-color: #00FFAA; color: #000; border: none; border-radius: 5px; }
    .stFileUploader { background-color: #1E1E1E; border: 1px solid #333; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# Define a storage path for uploaded PDFs
PDF_STORAGE_PATH = 'document_store/pdfs/'
os.makedirs(PDF_STORAGE_PATH, exist_ok=True)

# =============================================================================
# HELPER FUNCTION DEFINITIONS
# =============================================================================
def save_uploaded_file(uploaded_file):
    """
    Save the uploaded PDF file to disk.
    
    Args:
        uploaded_file: The file object provided by Streamlit file uploader.
        
    Returns:
        The file path where the PDF is saved.
    """
    file_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    """
    Load PDF documents from the given file path using PDFPlumberLoader.
    
    Args:
        file_path: Path to the saved PDF file.
    
    Returns:
        A list of document objects.
    """
    loader = PDFPlumberLoader(file_path)
    return loader.load()

def chunk_documents(raw_documents, chunk_size=1000, chunk_overlap=200):
    """
    Split raw documents into smaller text chunks.
    
    Args:
        raw_documents: The loaded documents.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.
        
    Returns:
        A list of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    return splitter.split_documents(raw_documents)

def index_documents(document_chunks):
    """
    Add document chunks to the persistent vector store.
    
    Args:
        document_chunks: List of document chunks.
    """
    st.session_state.vector_db.add_documents(document_chunks)

def find_related_documents(query, k=3):
    """
    Retrieve the top k relevant document chunks for the given query.
    
    Args:
        query: User query string.
        k: Number of top relevant chunks to retrieve.
    
    Returns:
        List of relevant document chunks.
    """
    return st.session_state.vector_db.similarity_search(query, k=k)

def generate_answer(user_query, context_documents):
    """
    Generate an answer using the selected local model by passing the query
    and its retrieved context to the language model.
    
    Args:
        user_query: The question provided by the user.
        context_documents: List of document chunks providing context.
        
    Returns:
        A tuple (answer, processing_time).
    """
    # Combine retrieved chunks into one context string
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    # Define the prompt template
    prompt_template = ChatPromptTemplate.from_template(
        """
You are an expert research assistant. Use the provided context to answer the query.
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query}
Context: {document_context}
Answer:
"""
    )
    # Initialize the local language model using the selected model
    language_model = OllamaLLM(model=st.session_state.selected_model)
    start_time = time.time()
    # Chain the prompt and language model
    response_chain = prompt_template | language_model
    answer = response_chain.invoke({
        "user_query": user_query,
        "document_context": context_text
    })
    processing_time = time.time() - start_time
    return answer, processing_time

# Asynchronous wrapper to offload the blocking generate_answer call
async def async_generate_answer(user_query, context_documents):
    return await asyncio.to_thread(generate_answer, user_query, context_documents)

# =============================================================================
# MAIN APPLICATION LOGIC
# =============================================================================
def main():
    # ---------------------------
    # SIDEBAR: Model Selection
    # ---------------------------
    st.sidebar.header("Local Model Selection")
    model_options = [
        "llama3:latest",
        "stable-code:latest",
        "gemma2:2b",
        "phi3:latest",
        "deepseek-r1:1.5b"
    ]
    selected_model = st.sidebar.selectbox(
        "Choose your Ollama model", model_options,
        help="Select one of your locally installed Ollama models"
    )
    
    # Persist or reinitialize the vector store if the model selection changes
    if "selected_model" not in st.session_state or st.session_state.selected_model != selected_model:
        st.session_state.selected_model = selected_model
        embedding_model = OllamaEmbeddings(model=selected_model)
        st.session_state.vector_db = InMemoryVectorStore(embedding_model)
        st.session_state.documents_indexed = False

    # ---------------------------
    # PAGE HEADER
    # ---------------------------
    st.title("📚 PDF-Chat-Wizard")
    st.markdown("### Chat with your PDFs using your local Ollama models")
    st.markdown("---")

    # ---------------------------
    # FILE UPLOAD SECTION
    # ---------------------------
    uploaded_files = st.file_uploader(
        "Upload one or more PDF documents",
        type="pdf",
        accept_multiple_files=True,
        help="Select PDF documents for analysis"
    )
    
    if uploaded_files:
        st.subheader("Uploaded Documents")
        all_chunks = []
        for uploaded_file in uploaded_files:
            st.write(f"**{uploaded_file.name}**")
            file_path = save_uploaded_file(uploaded_file)
            raw_docs = load_pdf_documents(file_path)
            # Display a preview of the first 500 characters of the first page
            if raw_docs:
                preview = raw_docs[0].page_content[:500]
                st.text_area("Preview", preview, height=150)
            chunks = chunk_documents(raw_docs)
            all_chunks.extend(chunks)
        # Index the document chunks only once per session unless new files are uploaded
        if not st.session_state.documents_indexed:
            with st.spinner("Indexing documents..."):
                index_documents(all_chunks)
            st.session_state.documents_indexed = True
            st.success("Documents processed and indexed successfully!")

    # ---------------------------
    # CHAT SECTION
    # ---------------------------
    st.subheader("Ask a Question About the Documents")
    user_query = st.text_input("Enter your query here:")
    
    if user_query:
        with st.spinner("Searching for relevant information asynchronously..."):
            # Retrieve only the top 3 most relevant chunks to limit the context size
            related_docs = find_related_documents(user_query, k=3)
            # Asynchronously generate the answer by offloading to a background thread
            answer, proc_time = asyncio.run(async_generate_answer(user_query, related_docs))
        st.markdown(f"**Answer:** {answer}")
        st.markdown(f"*Processed in {proc_time:.2f} seconds*")

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    main()
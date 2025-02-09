# PDF-Chat-Wizard

PDF-Chat-Wizard is a local PDF chatbot web application built using Streamlit, LangChain, and locally hosted Ollama language models. It enables users to upload PDF documents, automatically processes and indexes their content using an in-memory vector store, and provides concise, factual answers to technical queries via asynchronous inference.

## Overview

PDF-Chat-Wizard is designed for researchers, developers, and data scientists who need a fast, local tool for document analysis and query-based information retrieval. The application uses a multi-step pipeline:
- **PDF Extraction:** Uses PDFPlumberLoader to extract text from uploaded PDFs.
- **Text Chunking:** Splits the document into overlapping text chunks using a recursive text splitter.
- **Vector Indexing:** Converts chunks into embeddings with Ollama models and stores them in an in-memory vector store.
- **Query Retrieval:** Performs a similarity search to retrieve the most relevant chunks.
- **Asynchronous Inference:** Uses asynchronous processing to generate responses without blocking the UI.

## Features

- **Local Model Selection:**  
  Dynamically switch between multiple locally installed Ollama models via the sidebar:
  - `llama3:latest`
  - `stable-code:latest`
  - `gemma2:2b`
  - `phi3:latest`
  - `deepseek-r1:1.5b`

- **Multi-PDF Upload:**  
  Upload one or more PDF documents simultaneously.

- **Advanced Document Processing:**  
  Documents are parsed, chunked, and indexed for efficient similarity-based retrieval.

- **Asynchronous Inference:**  
  Inference calls are offloaded to background threads using `asyncio.to_thread`, keeping the UI responsive.

- **Optimized Context Retrieval:**  
  Only the top relevant document chunks are retrieved and used as context, minimizing input size and inference latency.

- **Professional and Responsive UI:**  
  Built with Streamlit and custom CSS, providing a modern, intuitive interface.

## Architecture & Technical Details

PDF-Chat-Wizard is organized into modular components:

1. **PDF Processing:**  
   Uploaded PDFs are saved to a designated directory and processed using PDFPlumberLoader. The extracted text is split into overlapping chunks using the `RecursiveCharacterTextSplitter`.

2. **Vector Store & Embeddings:**  
   Chunks are transformed into embeddings via a selected Ollama model (using `OllamaEmbeddings`) and stored in an in-memory vector store (`InMemoryVectorStore`). The vector store is cached in `st.session_state` to avoid re-indexing on every query.

3. **Similarity Search & Context Limitation:**  
   A similarity search retrieves the top 3 relevant chunks, keeping the context concise and reducing model input size.

4. **Prompt Chaining & Inference:**  
   A `ChatPromptTemplate` formats the user query along with the retrieved context. The combined prompt is then passed to an instance of `OllamaLLM` to generate the response. The entire process is wrapped in an asynchronous function to maintain UI responsiveness.

5. **Asynchronous Execution:**  
   Blocking model calls are offloaded using `asyncio.to_thread`, which allows for potential batch processing and concurrent query handling.

## Requirements

- **Python:** 3.8 or higher  
- **Streamlit:** For building the interactive web interface  
- **LangChain Libraries:** (PDFPlumberLoader, RecursiveCharacterTextSplitter, InMemoryVectorStore, ChatPromptTemplate)  
- **Ollama:** For local model hosting and inference (ensure your desired models are running locally)

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/pdf-chat-wizard.git
   cd pdf-chat-wizard
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install and Run Ollama Models:**

   Make sure you have [Ollama](https://ollama.com/) installed. Then, launch your desired models locally using the following commands in your terminal:

   - For **llama3:latest**:
     ```bash
     ollama run llama3:latest
     ```
   - For **stable-code:latest**:
     ```bash
     ollama run stable-code:latest
     ```
   - For **gemma2:2b**:
     ```bash
     ollama run gemma2:2b
     ```
   - For **phi3:latest**:
     ```bash
     ollama run phi3:latest
     ```
   - For **deepseek-r1:1.5b**:
     ```bash
     ollama run deepseek-r1:1.5b
     ```

   Ensure that the model you intend to use is running before launching the PDF-Chat-Wizard app.

## Usage

1. **Run the Application:**

   Launch the web app using Streamlit:

   ```bash
   streamlit run app.py
   ```

2. **Upload PDFs:**

   - Use the file uploader to select one or more PDF documents.
   - A preview (first 500 characters) of the document is displayed for verification.
   - The documents are processed and indexed only once (cached in session state) to speed up subsequent queries.

3. **Select a Local Model:**

   - Use the sidebar to select your preferred Ollama model.
   - Changing the model reinitializes the vector store, ensuring that embeddings are generated using the chosen model.

4. **Ask a Query:**

   - Enter your question in the “Ask a Question About the Documents” section.
   - The system retrieves the top 3 relevant document chunks and asynchronously generates an answer.
   - The response and processing time are displayed on the UI.

5. **Monitor Performance:**

   - The application shows the total processing time for each query.
   - For further performance tuning, experiment with reducing the number of retrieved chunks or switching to a more efficient model.

## Detailed Technical Flow

1. **File Upload & Save:**  
   Uploaded PDF files are saved in the `document_store/pdfs/` directory.

2. **Text Extraction & Chunking:**  
   The PDFPlumberLoader extracts text from each PDF, and the RecursiveCharacterTextSplitter divides the text into chunks (up to 1000 characters with 200-character overlap).

3. **Vector Embedding & Indexing:**  
   The text chunks are converted into embeddings using the selected Ollama model and stored in an in-memory vector store for fast similarity search. This index is cached in `st.session_state`.

4. **Query Handling:**  
   On query submission, the system retrieves the top 3 relevant chunks. This focused context is then used to construct a prompt with the ChatPromptTemplate.

5. **Asynchronous Answer Generation:**  
   The heavy inference call is offloaded to a background thread using `asyncio.to_thread`, ensuring that the UI remains responsive during processing.

PDF-Chat-Wizard is engineered to be modular, scalable, and efficient—making it a powerful tool for local document analysis and natural language querying.
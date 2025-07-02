# RAG-based Chat Application for Autism Spectrum Disorder (ASD) Research

This project implements a Retrieval-Augmented Generation (RAG) system to create a chatbot capable of answering questions based on a collection of scientific research papers on Autism Spectrum Disorder (ASD). It leverages LangChain, LangGraph, Perplexity AI, and Hugging Face embeddings to provide accurate, context-aware responses from a specialized knowledge base.

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Features

- **Automated Text Extraction**: Ingests research papers in PDF format.
- **Advanced Text Cleaning**: A dedicated script (`clean_text.py`) preprocesses the extracted text to remove noise such as headers, footers, citations, tables, and figure captions, which is crucial for the quality of the RAG system.
- **Vector-Based Retrieval**: Uses `sentence-transformers` for creating high-quality embeddings and an in-memory vector store for efficient similarity search.
- **Orchestrated RAG Pipeline**: Employs `LangGraph` to build a robust, stateful RAG pipeline, clearly defining the flow from retrieval to generation.
- **LLM Integration**: Uses the Perplexity API (`llama-3.1-sonar-small-128k-online`) for the generation step.
- **Traceability**: Integrated with LangSmith for debugging and monitoring the RAG pipeline.

## How It Works

The application follows a two-stage process: Data Preparation and RAG Pipeline Execution.

### 1. Data Preparation (`clean_text.py`)

Scientific PDFs have complex layouts that make direct text extraction noisy and unreliable for RAG systems. The `clean_text.py` script is designed to tackle this challenge:

1.  **PDF Parsing**: It reads PDF files from the `papers/` directory using `PyMuPDF`.
2.  **Noise Removal**: It applies a series of regular expressions and heuristics to remove common academic paper elements that are not part of the main content, such as:
    - In-text citations (e.g., `(Author, 2023)`, `[1]`).
    - Headers and footers containing journal names, page numbers, or dates.
    - Figure and table captions (e.g., `Fig. 1...`, `Table 2...`).
    - Content from tables, which often has a columnar format that disrupts sentence flow.
    - References and bibliography sections.
3.  **Output**: The cleaned, content-focused text is saved to the `cleaned_texts/` directory, providing a high-quality corpus for the RAG system.

### 2. RAG Pipeline (`main.py`)

The core RAG logic is orchestrated by a `LangGraph` state machine:

1.  **Load & Split**: The cleaned text files from `cleaned_texts/` are loaded and split into smaller, overlapping chunks.
2.  **Embed & Store**: Each chunk is converted into a vector embedding and stored in an in-memory `InMemoryVectorStore`.
3.  **Retrieve**: When a user asks a question (in `test.py`), the `retrieve` node searches the vector store for the most semantically similar document chunks.
4.  **Generate**: The retrieved chunks are combined with the original question and passed to the Perplexity LLM via a predefined prompt. The `generate` node produces a final, context-aware answer.

## Project Structure

```
RAG-langchain/
├── papers/                   # Input directory for raw PDF research papers
├── cleaned_texts/            # Output directory for cleaned text files
├── .env                      # For storing API keys (create this yourself)
├── clean_text.py             # Script for preprocessing and cleaning PDFs
├── main.py                   # Main application: sets up the RAG pipeline with LangGraph
├── test.py                   # A simple script to invoke the RAG graph and ask a question
└── README.md                 # This file
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd RAG-langchain
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows, use: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    You will need to create a `requirements.txt` file. Here are the likely dependencies:
    ```
    pip install python-dotenv langchain langgraph langchain-community langchain-huggingface langchain-perplexity sentence-transformers pymupdf pdf2image pytesseract
    ```
    *(Note: `pytesseract` may require installing the Tesseract-OCR engine on your system.)*

4.  **Set up environment variables:**
    Create a `.env` file in the root of the project and add your API keys:
    ```
    LANGSMITH_API_KEY="your_langsmith_api_key"
    PPLX_API_KEY="your_perplexity_api_key"
    ```

## Usage

1.  **Add Research Papers**:
    Place the PDF files of the research papers you want to include in the knowledge base into the `papers/` directory.

2.  **Clean the Text**:
    Run the text cleaning script. This will process the PDFs in `papers/` and save the cleaned versions in `cleaned_texts/`.
    ```bash
    python clean_text.py
    ```

3.  **Ask a Question**:
    Open `test.py`, modify the `question` in the `graph.invoke()` call, and run the script to get an answer.
    ```bash
    python test.py
    ```

## Dependencies

- `langchain`, `langgraph`, `langchain-community`, `langchain-core`
- `langchain-huggingface` for embeddings
- `langchain-perplexity` for the LLM
- `python-dotenv` for managing environment variables
- `PyMuPDF` (`fitz`) for PDF text extraction
- `pdf2image`, `pytesseract` for OCR on image-based PDFs
- `sentence-transformers` for text embeddings



# LangChain RAG Demo

This project is a small Retrieval-Augmented Generation (RAG) example built with Python, LangChain, OpenAI, and Pinecone. It demonstrates how to ingest documents, create embeddings, store them in a vector database, and then answer questions using retrieved context.

## What this project does

The repository includes a simple end-to-end RAG workflow:

1. Load documents from a source file.
2. Split them into smaller chunks.
3. Generate vector embeddings with OpenAI.
4. Store those embeddings in a Pinecone index.
5. Retrieve relevant chunks for a user question.
6. Send the retrieved context to an LLM and generate an answer.

## Project structure

- [main.py](main.py) - Demonstrates three approaches:
  - direct LLM prompting without RAG
  - a manual retrieval chain without LCEL
  - a cleaner LCEL-based retrieval chain
- [ingestion.py](ingestion.py) - Ingests a document file into Pinecone.
- [ingestion_csv.py](ingestion_csv.py) - Alternative ingestion script for CSV-based input.
- [write.py](write.py) - Helper that appends generated outputs to [output.txt](output.txt).
- [rag_it_help.csv](rag_it_help.csv) - Sample data used for ingestion.
- [mediumblog1.txt](mediumblog1.txt) - Additional sample text document.
- [pyproject.toml](pyproject.toml) - Project dependencies and Python version requirements.

## Prerequisites

- Python 3.11 or newer
- An OpenAI API key
- A Pinecone account and index
- The package manager uv (recommended) or pip

## Installation

From the project root, install dependencies with:

```bash
uv sync
```

If you are not using uv, you can install the dependencies from [pyproject.toml](pyproject.toml) with pip.

## Environment variables

Create a .env file or export the following values before running the scripts:

```bash
OPENAI_API_KEY=your_openai_key
INDEX_NAME=your_pinecone_index_name
```

If your setup requires Pinecone credentials in the environment, make sure they are available as well.

## Running the project

### 1. Ingest documents

```bash
python ingestion.py
```

This loads the documents, splits them into chunks, creates embeddings, and stores them in Pinecone.

### 2. Run the RAG demo

```bash
python main.py
```

The script will:

- ask the model directly without context
- run a manual retrieval workflow
- run an LCEL-based retrieval workflow

The results are written to [output.txt](output.txt).

### 3. Optional CSV ingestion

```bash
python ingestion_csv.py
```

This version is useful when working with CSV-based knowledge sources.

## How the RAG flow works

The project uses a classic retrieval pattern:

1. A user question is sent to the retriever.
2. The retriever finds the most relevant document chunks from Pinecone.
3. Those chunks are combined into a context string.
4. The LLM receives the question plus the retrieved context.
5. The model answers using the provided context instead of relying only on its pretrained knowledge.

## Notes

- The example uses a small sample dataset, so it is intended for learning and experimentation.
- The LCEL version is the more production-friendly approach because it is more declarative and easier to compose.
- The repository includes both a manual implementation and a LangChain Expression Language implementation to compare the styles.

## Technologies used

- LangChain
- LangChain OpenAI integration
- Pinecone vector store
- Python dotenv
- Unstructured document loading
- OpenAI embeddings and chat models

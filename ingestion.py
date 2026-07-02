from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from pathlib import Path
import logging
import os
import json

logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()

if __name__ == "__main__":
    logging.info("Ingesting...")
    filepath = Path.cwd() / "Docs" / "mediumblog1.txt"
    logging.info(f"File path: {filepath}")
    loader = UnstructuredLoader(file_path=filepath, chunking_strategy="basic", max_characters=1000000)
    document = loader.load()
    logging.info(f"Loaded {len(document)} documents.")

    logging.info("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    logging.info(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    logging.info("ingesting...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    logging.info("finish")

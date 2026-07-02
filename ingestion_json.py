from dotenv import load_dotenv
from pathlib import Path
import json
import logging
import os

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
load_dotenv()

if __name__ == "__main__":

    logging.info("Loading JSON...")
    filepath = Path.cwd() / "Docs" / "rag_it_help_openai_langchain.json"
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    documents = []
    for item in data:
        doc = Document(
            page_content=item["page_content"],
            metadata={
                "id": item["id"],
                "topic": item["topic"],
                "question": item["question"],
                "source": item["metadata"]["source"],
                "row": item["metadata"]["row"],
            },
        )
        documents.append(doc)
    logging.info(f"Loaded {len(documents)} documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=[
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)
    logging.info(f"Created {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.environ["OPENAI_API_KEY"],
    )

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=os.environ["INDEX_NAME"],
    )
    logging.info("Finished!")

from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from pathlib import Path
import logging
import os
import json

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()

if __name__ == "__main__":
    logging.info("Ingesting...")
    filepath = Path.cwd() / "rag_it_help.csv"
    logging.info(f"File path: {filepath}")
    loader = loader = UnstructuredLoader(
        file_path=filepath,
        chunking_strategy="basic",
        max_characters=1000,
        new_after_n_chars=800,
    )
    document = loader.load()
    logging.info(f"Loaded {len(document)} documents.")
    texts = document

    # for remove metadata keys .csv file has a lot of metadata keys, we will keep only filename and source
    for doc in texts:
        doc.metadata = {
            "source": doc.metadata.get("source", str(filepath)),
            "filename": doc.metadata.get("filename", filepath.name),
        }
    # Enable if metadata consumes a lot of space in the console and pinecone index
    # for i, doc in enumerate(texts[:3]):
    #     print(f"Document {i}")
    #     print(doc.metadata.keys())
    #     print("Metadata size:", len(json.dumps(doc.metadata)))

    # import json

    # for k, v in texts[0].metadata.items():
    #     try:
    #         size = len(json.dumps(v))
    #     except Exception:
    #         size = len(str(v))
    #     print(f"{k:30} {size}")

    # print(len(texts[0].page_content))

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    logging.info("ingesting...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    logging.info("finish")

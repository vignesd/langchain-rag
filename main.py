import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pathlib import Path
from write import write_data_to_file

import logging
logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()

logging.info("Initializing components...")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0)
vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


# ============================================================================
# IMPLEMENTATION 1: Without LCEL (Simple Function-Based Approach)
# ============================================================================
def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query)
    # Step 2: Format documents into context string
    context = format_docs(docs)
    # Step 3: Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)
    # Step 4: Invoke LLM with the formatted messages
    response = llm.invoke(messages)
    # Step 5: Return the content
    return response.content


# ============================================================================
# IMPLEMENTATION 2: With LCEL (LangChain Expression Language) - BETTER APPROACH
# ============================================================================
def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


if __name__ == "__main__":
    logging.info("Retrieving...")
    filepath = Path.cwd() / "output.txt"

    # Query
    # query = "what is Pinecone in machine learning?"
    query ="Setting Up a Mobile Device for Company Email"

    # ========================================================================
    # Option 0: Raw invocation without RAG
    # ========================================================================
    logging.info("\n" + "=" * 70)
    logging.info("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    logging.info("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    logging.info("\nAnswer:")
    logging.info(result_raw.content)
    write_data_to_file("Raw LLM Invocation (No RAG)", result_raw.content, filepath)


    # ========================================================================
    # Option 1: Use implementation WITHOUT LCEL
    # ========================================================================
    logging.info("\n" + "=" * 70)
    logging.info("IMPLEMENTATION 1: Without LCEL")
    logging.info("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    logging.info("\nAnswer:")
    logging.info(result_without_lcel)
    write_data_to_file("WITHOUT LCEL", result_without_lcel, filepath)

    # # ========================================================================
    # # Option 2: Use implementation WITH LCEL (Better Approach)
    # # ========================================================================
    logging.info("\n" + "=" * 70)
    logging.info("IMPLEMENTATION 2: With LCEL - Better Approach")
    logging.info("=" * 70)
    logging.info("Why LCEL is better:")
    logging.info("- More concise and declarative")
    logging.info("- Built-in streaming: chain.stream()")
    logging.info("- Built-in async: chain.ainvoke()")
    logging.info("- Easy to compose with other chains")
    logging.info("- Better for production use")
    logging.info("=" * 70)
    

    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    logging.info("\nAnswer:")
    logging.info(result_with_lcel)
    write_data_to_file("With LCEL", result_with_lcel, filepath)

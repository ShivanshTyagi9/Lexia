from utils.retriver import get_multiquery_retriever
from utils.rerank import rerank_chunks
from utils.vectorstore import load_vectorstore
from utils.parent_store import create_parent_retriever
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1", base_url="http://localhost:11434")

def retrieve_final_docs(question: str):
    # Load vectorstore for multiquery
    vectordb = load_vectorstore("data/chroma_db")
    multiquery = get_multiquery_retriever(vectordb, llm)

    child_chunks = multiquery.invoke(question)

    top_children = rerank_chunks(question, child_chunks, top_k=150, threshold=0.1)

    parent_retriever = create_parent_retriever(top_children, persist_directory="data/chroma_parent_db")
    parent_docs = parent_retriever.invoke(question)

    top_parents = rerank_chunks(question, parent_docs, top_k=30, threshold=0.1)
    return top_parents

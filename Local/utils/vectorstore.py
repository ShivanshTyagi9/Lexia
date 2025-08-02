from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

def store_chunks_in_chroma(chunks, persist_directory="data/chroma_db"):
    embedding = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        collection_name="multi-pdf",
        persist_directory=persist_directory
    )
    vectordb.persist()
    return vectordb

def load_vectorstore(persist_directory="data/chroma_db"):
    embedding = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    return Chroma(
        embedding_function=embedding,
        collection_name="multi-pdf",
        persist_directory=persist_directory
    )

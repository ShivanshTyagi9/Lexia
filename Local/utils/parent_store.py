from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings


def split_into_parent_and_child_chunks(docs):
    
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)

    parent_docs = parent_splitter.split_documents(docs)
    return parent_docs, child_splitter

"""
def create_parent_retriever(docs, persist_directory="data/chroma_parent_db"):
    parent_docs, child_splitter = split_into_parent_and_child_chunks(docs)

    # Use Ollama local embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

    # Vector store holds child chunks
    vectorstore = Chroma(
        collection_name="parent-child",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    # Store full parent chunks (in-memory or Redis/FAISS if needed)
    docstore = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
    )
    parent_docs = [doc for doc in parent_docs if doc.page_content.strip()]
    # Add docs to retriever
    retriever.add_documents(parent_docs)
    vectorstore.persist()

    return retriever
"""

def create_parent_retriever(docs, persist_directory="data/chroma_parent_db"):
    parent_docs, child_splitter = split_into_parent_and_child_chunks(docs)
    
    parent_docs = [doc for doc in parent_docs if isinstance(doc.page_content, str) and doc.page_content.strip()]

    # Create embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

    # Instead of letting retriever handle .add_documents(), we manually embed + insert
    texts = [doc.page_content for doc in parent_docs]
    metadatas = [doc.metadata for doc in parent_docs]
    docstore = InMemoryStore()
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name="parent-child",
        persist_directory=persist_directory
    )

    # Manually store the full parent docs in the docstore
    docstore.mset([(f"doc-{i}", doc) for i, doc in enumerate(parent_docs)])

    # Set up retriever using existing vectorstore and docstore
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        search_kwargs={"k": 40}  # optional
    )

    return retriever

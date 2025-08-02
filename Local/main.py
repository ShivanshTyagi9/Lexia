from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
from utils.loader import load_all_docs_from_directory
from utils.vectorstore import store_chunks_in_chroma, load_vectorstore
from utils.retriver import get_multiquery_retriever
from utils.chain import build_rag_chain
from utils.ingestion_log import update_log
from utils.parent_store import create_parent_retriever
from utils.rerank import rerank_chunks
from utils.combined_query import retrieve_final_docs
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
#from utils.chain import build_rag_chain_with_context
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PDF_DIR = Path("data/pdfs")
CHROMA_DIR = "data/chroma_db"
PDF_DIR.mkdir(parents=True, exist_ok=True)

llm = ChatOllama(model="wizard-vicuna-uncensored:7b", base_url="http://localhost:11434")
#llm = ChatGoogleGenerativeAI(temperature = 0.7,model="gemini-2.0-flash-lite")
vectordb = None
rag_chain = None

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    filepath = PDF_DIR / file.filename
    with open(filepath, "wb") as f:
        f.write(await file.read())
    return {"filename": file.filename, "message": "Uploaded successfully."}


"""
@app.post("/ingest")
def ingest_documents():
    chunks = load_all_docs_from_directory(PDF_DIR)
    if not chunks:
        return JSONResponse(status_code=400, content={"message": "No documents to ingest."})
    new_files = {chunk.metadata["source"] for chunk in chunks}
    update_log(new_files)
    global vectordb, rag_chain
    vectordb = store_chunks_in_chroma(chunks, persist_directory=CHROMA_DIR)
    retriever = get_multiquery_retriever(vectordb, llm)
    rag_chain = build_rag_chain(retriever, llm)
    return {"message": f"Ingested {len(chunks)} chunks from PDFs."}
"""

"""
@app.post("/ingest")
def ingest_documents():
    chunks = load_all_docs_from_directory(PDF_DIR)
    if not chunks:
        return JSONResponse(status_code=400, content={"message": "No documents to ingest."})
    new_files = {chunk.metadata["source"] for chunk in chunks}
    update_log(new_files)

    global rag_chain
    # Build parent retriever and chain
    retriever = create_parent_retriever(chunks, persist_directory="data/chroma_parent_db")
    rag_chain = build_rag_chain(retriever, llm)
    return {"message": f"Ingested {len(chunks)} chunks and built parent retriever."}
"""
"""
@app.post("/ingest")
def ingest_documents():
    chunks = load_all_docs_from_directory(PDF_DIR)
    if not chunks:
        return JSONResponse(status_code=400, content={"message": "No documents to ingest."})
    new_files = {chunk.metadata["source"] for chunk in chunks}
    update_log(new_files)

    global retriever
    retriever = create_parent_retriever(chunks, persist_directory="data/chroma_parent_db")
    return {"message": f"Ingested {len(chunks)} chunks and built parent retriever."}
"""


@app.post("/ingest")
def ingest_documents():
    chunks = load_all_docs_from_directory(PDF_DIR)
    if not chunks:
        return JSONResponse(status_code=400, content={"message": "No documents to ingest."})

    new_files = {chunk.metadata["source"] for chunk in chunks}
    update_log(new_files)

    # Save child chunks into Chroma vector store for multi-query
    store_chunks_in_chroma(chunks, persist_directory="data/chroma_db")

    # Create parent retriever for mapping child → parent
    global parent_retriever
    parent_retriever = create_parent_retriever(chunks, persist_directory="data/chroma_parent_db")

    return {"message": f"Ingested {len(chunks)} chunks and built vector + parent retriever."}

@app.get("/documents")
def list_documents():
    #if not vectordb:
    vectordb = load_vectorstore(CHROMA_DIR)
    return {"documents": list(set(d.metadata['source'] for d in vectordb.similarity_search("test", k=1)))}

@app.get("/query")
def query_documents(question: str = Query(...)):
    top_docs = retrieve_final_docs(question)
    rag_chain = build_rag_chain(lambda _: top_docs, llm)
    result = rag_chain.invoke(question)
    return {"answer": result}


"""
@app.get("/query")
def query_documents(question: str = Query(...)):
    global vectordb, rag_chain
    if not rag_chain:
        vectordb = load_vectorstore(CHROMA_DIR)
        retriever = get_multiquery_retriever(vectordb, llm)
        rag_chain = build_rag_chain(retriever, llm)
    result = rag_chain.invoke(question)
    return {"answer": result}
"""
"""
@app.get("/query")
def query_documents(question: str = Query(...)):
    global retriever, rag_chain
    if not retriever:
        chunks = load_all_docs_from_directory(PDF_DIR)
        retriever = create_parent_retriever(chunks, persist_directory="data/chroma_parent_db")

    raw_docs = retriever.invoke(question)
    top_docs = rerank_chunks(question, raw_docs, top_k=30, threshold=0.1)

    # You can reuse your existing chain template
    rag_chain = build_rag_chain(lambda _: top_docs, llm)
    result = rag_chain.invoke(question)
    return {"answer": result}
"""
"""
@app.get("/query")
def query_documents(question: str = Query(...)):
    global rag_chain
    if not rag_chain:
        chunks = load_all_docs_from_directory(PDF_DIR)
        retriever = create_parent_retriever(chunks, persist_directory="data/chroma_parent_db")
        rag_chain = build_rag_chain(retriever, llm)

    result = rag_chain.invoke(question)
    return {"answer": result}
"""
"""
@app.get("/query")
def query_documents(question: str = Query(...)):
    global rag_chain, vectordb

    if not rag_chain:
        vectordb = load_vectorstore(CHROMA_DIR)
        retriever = get_multiquery_retriever(vectordb, llm)
        rag_chain = build_rag_chain_with_context(retriever, llm)

    result = rag_chain(question)
    return {
        "answer": result["answer"],
        "contexts": result["contexts"]
    }
"""
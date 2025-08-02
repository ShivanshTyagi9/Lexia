from pathlib import Path
from langchain_community.document_loaders import UnstructuredPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.ingestion_log import load_log


def load_all_docs_from_directory(doc_dir):
    doc_dir = Path(doc_dir)
    all_chunks = []
    already_ingested = load_log()
    
    for file in doc_dir.glob("*"):
        if file.name in already_ingested:
            continue

        if file.suffix.lower() == ".pdf":
            loader = UnstructuredPDFLoader(str(file))
        elif file.suffix.lower() == ".txt":
            loader = TextLoader(str(file), encoding="utf-8")
        else:
            continue

        try:
            data = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(data)
            for chunk in chunks:
                context_header = f"[{file.name}]"
                chunk.page_content = f"{context_header}\n{chunk.page_content}"
                chunk.metadata["source"] = file.name
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Failed to load {file.name}: {e}")
    
    return all_chunks

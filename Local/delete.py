import shutil
from pathlib import Path

def clear_chroma_db(persist_path="data/chroma_parent_db"):
    db_dir = Path(persist_path)
    if db_dir.exists() and db_dir.is_dir():
        shutil.rmtree(db_dir)
        print(f"✅ Cleared Chroma DB at: {persist_path}")
    else:
        print(f"⚠️ No Chroma DB found at: {persist_path}")

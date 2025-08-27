import json
from pathlib import Path

LOG_PATH = Path("data/ingested_files.json")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_log():
    if LOG_PATH.exists():
        return set(json.loads(LOG_PATH.read_text(encoding="utf-8")))
    return set()

def update_log(new_files):
    log = load_log()
    updated = log.union(new_files)
    LOG_PATH.write_text(json.dumps(list(updated), ensure_ascii=False, indent=2), encoding="utf-8")
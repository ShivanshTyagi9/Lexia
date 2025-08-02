import json
from pathlib import Path

LOG_PATH = Path("data/ingested_files.json")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_log():
    if LOG_PATH.exists():
        with open(LOG_PATH, "r") as f:
            return set(json.load(f))
    return set()

def update_log(new_files):
    log = load_log()
    updated = log.union(new_files)
    with open(LOG_PATH, "w") as f:
        json.dump(list(updated), f)

import json
import os

DATA_DIR = "./data"
STATE_FILE = os.path.join(DATA_DIR, "state.json")

os.makedirs(DATA_DIR, exist_ok=True)

def _read_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _write_state(data: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def get_trained_files_list() -> list:
    data = _read_state()
    return data.get("trained_files", [])

def add_trained_files_to_list(filenames: list):
    data = _read_state()
    current = set(data.get("trained_files", []))
    for f in filenames:
        current.add(f)
    data["trained_files"] = list(current)
    _write_state(data)

def clear_trained_files_list():
    data = _read_state()
    data["trained_files"] = []
    _write_state(data)

def remove_trained_file_from_list(filename: str):
    data = _read_state()
    current = data.get("trained_files", [])
    if filename in current:
        current.remove(filename)
        data["trained_files"] = current
        _write_state(data)

import json
import os

SETTINGS_FILE = "settings.json"

DEFAULT_TEMPLATE = """Jesteś inteligentnym i pomocnym asystentem AI.
Został Ci dostarczony poniższy KONTEKST w postaci fragmentów dokumentów.
Odpowiedz na pytanie bazując na tym kontekście. Odpowiedz po polsku.
Jeśli nie potrafisz znaleźć odpowiedzi w kontekście, powiedz o tym, a następnie odpowiedz zgodnie z własną wiedzą.

KONTEKST:
{context}

PYTANIE UŻYTKOWNIKA:
{question}"""

def _read_settings() -> dict:
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _write_settings(data: dict) -> None:
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def get_saved_template() -> str:
    data = _read_settings()
    return data.get("template", DEFAULT_TEMPLATE)

def save_template(template: str) -> None:
    data = _read_settings()
    data["template"] = template
    _write_settings(data)

# --- NOWE FUNKCJE DLA PAMIĘCI RAG ---

def get_trained_files_list() -> list:
    data = _read_settings()
    return data.get("trained_files", [])

def add_trained_files_to_list(filenames: list):
    data = _read_settings()
    current = set(data.get("trained_files", []))
    for f in filenames:
        current.add(f)
    data["trained_files"] = list(current)
    _write_settings(data)

def clear_trained_files_list():
    data = _read_settings()
    data["trained_files"] = []
    _write_settings(data)

def remove_trained_file_from_list(filename: str):
    data = _read_settings()
    current = data.get("trained_files", [])
    if filename in current:
        current.remove(filename)
        data["trained_files"] = current
        _write_settings(data)

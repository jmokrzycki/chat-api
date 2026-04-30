import json
import os

DATA_DIR = "./data"
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")

os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_TEMPLATE = """Jesteś inteligentnym i pomocnym asystentem AI. Poniżej znajduje się historia Twojej rozmowy z Użytkownikiem.
Twoim zadaniem jest odpowiedzieć na najnowsze PYTANIE UŻYTKOWNIKA bazując na podanym KONTEKŚCIE z dokumentów oraz HISTORII ROZMOWY.

ZASADY:
1. Pamiętaj, kim jesteś (Asystentem) i z kim rozmawiasz (Użytkownik). Nie przejmuj tożsamości Użytkownika.
2. Jeśli kontekst z dokumentów jest pusty, oprzyj się na historii rozmowy.
3. Odpowiadaj zawsze w języku polskim, zwięźle i naturalnie.

HISTORIA ROZMOWY (Może być pusta):
{chat_history}

KONTEKST Z DOKUMENTÓW:
{context}

PYTANIE UŻYTKOWNIKA:
{question}"""

DEFAULT_REPHRASE_TEMPLATE = """Jesteś ekspertem od NLP. Twoim zadaniem jest stworzenie JEDNEGO, samodzielnego zapytania wyszukiwania (Standalone Question).
Przeanalizuj HISTORIĘ ROZMOWY oraz NAJNOWSZE PYTANIE.

Jeśli najnowsze pytanie nawiązuje do historii (np. ma zaimki "to", "on" lub odnosi się do imienia Użytkownika, np. "jak mam na imię?"), zamień je na obiektywne zapytanie (np. "Jakie jest imię użytkownika?").

ZASADY:
1. Nie odpowiadaj na pytanie!
2. Nie używaj zwrotów grzecznościowych (Cześć, witaj).
3. Jeśli pytanie jest już jasne samo w sobie, skopiuj je bez zmian.
4. Zwróć TYLKO przebudowane zapytanie.

HISTORIA ROZMOWY:
{chat_history}

NAJNOWSZE PYTANIE:
{question}

SAMODZIELNE ZAPYTANIE DO BAZY DANYCH:"""

DEFAULT_HISTORY_LIMIT = 4
DEFAULT_MEMORY_ENABLED = True

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

def get_settings_data() -> dict:
    data = _read_settings()
    return {
        "template": data.get("template", DEFAULT_TEMPLATE),
        "rephrase_template": data.get("rephrase_template", DEFAULT_REPHRASE_TEMPLATE),
        "history_limit": data.get("history_limit", DEFAULT_HISTORY_LIMIT),
        "memory_enabled": data.get("memory_enabled", DEFAULT_MEMORY_ENABLED)
    }

def save_settings_data(template: str, rephrase_template: str, history_limit: int, memory_enabled: bool) -> None:
    data = _read_settings()
    data["template"] = template
    data["rephrase_template"] = rephrase_template
    data["history_limit"] = history_limit
    data["memory_enabled"] = memory_enabled
    _write_settings(data)

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
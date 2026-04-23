import json
import os

SETTINGS_FILE = "settings.json"

DEFAULT_TEMPLATE = """Jesteś inteligentnym i pomocnym asystentem AI.
Został Ci dostarczony poniższy KONTEKST w postaci fragmentów dokumentów.
Odpowiedz na pytanie bazując na tym kontekście.
Jeśli nie potrafisz znaleźć odpowiedzi w kontekście, powiedz o tym, a następnie odpowiedz zgodnie z własną wiedzą.

KONTEKST:
{context}

PYTANIE UŻYTKOWNIKA:
{question}"""

def get_saved_template() -> str:
    if not os.path.exists(SETTINGS_FILE):
        return DEFAULT_TEMPLATE
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("template", DEFAULT_TEMPLATE)
    except Exception:
        return DEFAULT_TEMPLATE

def save_template(template: str) -> None:
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump({"template": template}, f, ensure_ascii=False, indent=4)
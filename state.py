"""
state.py — Завантаження та збереження стану індексації між перезапусками.

ВАЖЛИВО: інші модулі повинні звертатись до стану ВИКЛЮЧНО через функції
get_indexed_states() / get_doc_summaries(), а не імпортувати словники напряму.
Пряме `from state import _indexed_states` дає копію на момент імпорту і
не відображає оновлення після load_persistent_state().
"""

import json
import logging

from config import STATES_FILE, SUMMARIES_FILE

log = logging.getLogger("state")

# ── Внутрішній стан (не імпортувати напряму!) ─────────────────────────────────
_indexed_states: dict = {}  # {filename: mtime}
_doc_summaries:  dict = {}  # {filename: {...}}


# ── Геттери (використовувати в інших модулях) ─────────────────────────────────

def get_indexed_states() -> dict:
    """Повертає живе посилання на словник стану індексації."""
    return _indexed_states


def get_doc_summaries() -> dict:
    """Повертає живе посилання на словник резюме документів."""
    return _doc_summaries


# ── Завантаження / збереження ─────────────────────────────────────────────────

def load_persistent_state() -> None:
    """Завантажити збережений стан індексації та резюме документів з диска."""
    global _indexed_states, _doc_summaries

    if STATES_FILE.exists():
        try:
            loaded = json.loads(STATES_FILE.read_text(encoding="utf-8"))
            # Оновлюємо існуючий об'єкт — всі модулі, що тримають посилання, побачать зміни
            _indexed_states.clear()
            _indexed_states.update(loaded)
            log.info("Завантажено стан індексації: %d файлів", len(_indexed_states))
        except Exception as exc:
            log.error("Не вдалося прочитати index_states.json: %s", exc)
            _indexed_states.clear()
    else:
        log.info("index_states.json не знайдено — починаємо з порожнього стану")

    if SUMMARIES_FILE.exists():
        try:
            loaded = json.loads(SUMMARIES_FILE.read_text(encoding="utf-8"))
            _doc_summaries.clear()
            _doc_summaries.update(loaded)
            log.info("Завантажено резюме: %d документів", len(_doc_summaries))
        except Exception as exc:
            log.error("Не вдалося прочитати doc_summaries.json: %s", exc)
            _doc_summaries.clear()
    else:
        log.info("doc_summaries.json не знайдено — починаємо з порожнього стану")


def save_persistent_state() -> None:
    """Записати поточний стан на диск."""
    try:
        STATES_FILE.write_text(
            json.dumps(_indexed_states, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        SUMMARIES_FILE.write_text(
            json.dumps(_doc_summaries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.debug("Стан збережено на диск (%d файлів).", len(_indexed_states))
    except Exception as exc:
        log.error("Помилка збереження стану: %s", exc)
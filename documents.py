"""
documents.py — Допоміжні функції для роботи з документами сторін та справи.
"""

import datetime
import logging
import re
from pathlib import Path

from fastapi import UploadFile

from ocr import extract_text

log = logging.getLogger("documents")


# ── Резюме документа ──────────────────────────────────────────────────────────

def build_doc_summary(filename: str, text: str, chunk_count: int) -> dict:
    """
    Формує стислу характеристику документа:
    кількість символів, слів, рядків, мова (евристика), перший блок тексту.
    """
    lines = [l for l in text.splitlines() if l.strip()]
    words = text.split()
    chars = len(text)

    sample = text[:200].lower()
    if any(c in sample for c in "іїєґ"):
        lang = "uk"
    elif any(c in sample for c in "ыъэё"):
        lang = "ru"
    else:
        lang = "en/other"

    preview_raw = text[:800].strip()
    last_dot = max(
        preview_raw.rfind(". "),
        preview_raw.rfind(".\n"),
        preview_raw.rfind("! "),
        preview_raw.rfind("? "),
    )
    preview = preview_raw[:last_dot + 1] if last_dot > 100 else preview_raw[:600]

    headings: list[str] = []
    for line in lines[:30]:
        stripped = line.strip()
        if (stripped.isupper() and 5 < len(stripped) < 80) or \
           (len(stripped) < 80 and stripped and stripped[0].isdigit()):
            headings.append(stripped[:80])
        if len(headings) >= 8:
            break

    return {
        "filename":    filename,
        "char_count":  chars,
        "word_count":  len(words),
        "line_count":  len(lines),
        "chunk_count": chunk_count,
        "language":    lang,
        "preview":     preview,
        "headings":    headings,
        "indexed_at":  datetime.datetime.now().isoformat(timespec="seconds"),
    }


# ── Читання файлу за шляхом ───────────────────────────────────────────────────

async def extract_text_from_path(file_path: str) -> tuple[str, bool, int]:
    """Зчитати файл за шляхом і повернути (текст, успіх, розмір_байт)."""
    path = Path(file_path)
    if not path.exists():
        log.warning("extract_text_from_path: файл не знайдено '%s'", file_path)
        return "", False, 0

    try:
        data = path.read_bytes()
        size = len(data)
        text = extract_text(data, path.name)
        log.debug("extract_text_from_path: '%s' → %d символів", file_path, len(text))
        return text, bool(text.strip()), size
    except Exception as exc:
        log.error("Не вдалося прочитати '%s': %s", file_path, exc)
        return "", False, 0


# ── Витяг документа справи ───────────────────────────────────────────────────

async def extract_case_doc(
    upload: UploadFile | None,
    label: str,
) -> tuple[str, dict | None]:
    """
    Витягти текст з документа справи (завантаженого файлу).
    Повертає: (текст_для_промпту, мета_для_фронтенду | None)
    """
    if upload is None or not upload.filename:
        return "", None

    data = await upload.read()
    if not data:
        log.warning("extract_case_doc: '%s' — порожній файл", upload.filename)
        return "", None

    text    = extract_text(data, upload.filename, upload.content_type or "")
    size_kb = round(len(data) / 1024, 1)

    if text.strip():
        chars     = len(text.strip())
        words     = len(text.split())
        ok        = True
        preview   = text.strip()[:500]
        full_text = f"[{label} — файл: {upload.filename}]\n{text.strip()}"
        log.info("extract_case_doc: '%s' (%s) — %d символів", upload.filename, label, chars)
    else:
        chars, words, ok = 0, 0, False
        preview   = "(Документ не містить розпізнаного тексту або пошкоджений)"
        full_text = f"[{label} — файл: {upload.filename}]\n{preview}"
        log.warning("extract_case_doc: '%s' (%s) — текст не розпізнано", upload.filename, label)

    meta = {
        "filename": upload.filename,
        "label":    label,
        "size_kb":  size_kb,
        "chars":    chars,
        "words":    words,
        "ok":       ok,
        "preview":  preview,
        "group":    "case",
    }
    return full_text, meta


# ── Збір тексту сторони ───────────────────────────────────────────────────────

async def collect_party_text(
    files:       list[UploadFile],
    raw_text:    str,
    party_label: str,
) -> tuple[str, list[dict]]:
    """
    Збираємо текст сторони + метадані кожного файлу.
    Повертає: (текст_для_промпту, список_мета_файлів)
    """
    parts:      list[str]  = []
    file_metas: list[dict] = []

    if raw_text.strip():
        parts.append(raw_text.strip())
        file_metas.append({
            "filename": "(текст введено вручну)",
            "label":    party_label,
            "size_kb":  0,
            "chars":    len(raw_text.strip()),
            "words":    len(raw_text.split()),
            "ok":       True,
            "preview":  raw_text.strip()[:500],
            "group":    party_label.lower(),
        })
        log.debug("collect_party_text (%s): додано текст вручну (%d символів)", party_label, len(raw_text))

    for uf in files:
        if not uf or not uf.filename:
            continue
        data      = await uf.read()
        if not data:
            continue
        extracted = extract_text(data, uf.filename, uf.content_type or "")
        size_kb   = round(len(data) / 1024, 1)

        if extracted.strip():
            parts.append(f"[Файл: {uf.filename}]\n{extracted.strip()}")
            file_metas.append({
                "filename": uf.filename,
                "label":    party_label,
                "size_kb":  size_kb,
                "chars":    len(extracted.strip()),
                "words":    len(extracted.split()),
                "ok":       True,
                "preview":  extracted.strip()[:500],
                "group":    party_label.lower(),
            })
            log.info("collect_party_text (%s): файл '%s' — %d символів", party_label, uf.filename, len(extracted))
        else:
            parts.append(
                f"[Файл: {uf.filename}]\n"
                "(Документ не містить розпізнаного тексту або пошкоджений)"
            )
            file_metas.append({
                "filename": uf.filename,
                "label":    party_label,
                "size_kb":  size_kb,
                "chars":    0,
                "words":    0,
                "ok":       False,
                "preview":  "(Не вдалося витягти текст — документ порожній або пошкоджений)",
                "group":    party_label.lower(),
            })
            log.warning("collect_party_text (%s): файл '%s' — текст не витягнуто", party_label, uf.filename)

    if not parts:
        log.info("collect_party_text (%s): жодних даних не надано", party_label)
        return (
            f"(Сторона «{party_label}» не надала жодних пояснень чи документів. "
            "Аналіз проводиться на основі даних іншої сторони)."
        ), []

    return "\n\n---\n\n".join(parts), file_metas

"""
ocr.py — Витягування тексту з PDF, DOCX, зображень та текстових файлів.
"""

import io
import logging
from pathlib import Path

import docx
import easyocr
import fitz
import numpy as np
from PIL import Image

log = logging.getLogger("ocr")

# ── Ініціалізація EasyOCR (один раз при запуску) ─────────────────────────────
log.info("Завантаження EasyOCR... (займе кілька секунд)")
ocr_reader = easyocr.Reader(["uk", "ru", "en"], gpu=False)
log.info("EasyOCR готовий.")


# ── PDF ───────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Витягнути текст з PDF: спочатку вбудований текст, потім OCR для відсканованих сторінок."""
    text_parts: list[str] = []
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for i, page in enumerate(doc):
            page_text = page.get_text()

            if not page_text or len(page_text.strip()) < 50:
                log.debug("PDF сторінка %d: замало тексту, спробуємо OCR", i + 1)
                try:
                    pix = page.get_pixmap(dpi=300)
                    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                    if pix.n == 4:
                        img_array = img_array[:, :, :3]
                    results = ocr_reader.readtext(img_array, detail=0)
                    ocr_text = "\n".join(results)
                    if ocr_text.strip():
                        page_text = ocr_text
                        log.debug("PDF сторінка %d: OCR вилучив %d символів", i + 1, len(ocr_text))
                except Exception as exc:
                    log.warning("PDF OCR — сторінка %d: %s", i + 1, exc)

            if page_text and page_text.strip():
                text_parts.append(page_text.strip())

    except Exception as exc:
        log.error("Помилка читання PDF: %s", exc)

    result = "\n".join(text_parts).strip()
    log.debug("PDF: витягнуто %d символів", len(result))
    return result


# ── Зображення ────────────────────────────────────────────────────────────────

def extract_text_from_image(file_bytes: bytes) -> str:
    """Розпізнати текст на зображенні через EasyOCR."""
    try:
        image = Image.open(io.BytesIO(file_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        img_array = np.array(image)
        results = ocr_reader.readtext(img_array, detail=0)
        text = "\n".join(results).strip()
        log.debug("Image OCR: витягнуто %d символів", len(text))
        return text
    except Exception as exc:
        log.error("Помилка Image OCR: %s", exc)
        return ""


# ── DOCX ──────────────────────────────────────────────────────────────────────

def extract_text_from_docx(file_bytes: bytes) -> str:
    """Зчитати текст з DOCX-документа."""
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        log.debug("DOCX: витягнуто %d символів", len(text))
        return text
    except Exception as exc:
        log.error("Помилка читання DOCX: %s", exc)
        return f"[DOCX] Помилка: {exc}"


# ── Універсальна функція ──────────────────────────────────────────────────────

def extract_text(file_bytes: bytes, filename: str, mime_type: str = "") -> str:
    """
    Визначити тип файлу та витягнути текст відповідним методом.
    Підтримує: PDF, DOCX, зображення, TXT/MD.
    """
    ext = Path(filename).suffix.lower()
    mt  = mime_type.lower()
    log.debug("extract_text: файл='%s', ext='%s', mime='%s'", filename, ext, mt)

    if ext == ".pdf" or "pdf" in mt:
        return extract_text_from_pdf(file_bytes)
    if ext == ".docx" or "wordprocessingml" in mt:
        return extract_text_from_docx(file_bytes)
    if ext in {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp"} or "image/" in mt:
        return extract_text_from_image(file_bytes)

    # Текстові файли (.txt, .md) — перебираємо кодування
    for enc in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        try:
            text = file_bytes.decode(enc)
            log.debug("Текстовий файл '%s' декодовано як %s (%d символів)", filename, enc, len(text))
            return text
        except (UnicodeDecodeError, LookupError):
            continue

    log.warning("Файл '%s': не вдалося визначити кодування, використовуємо latin-1 з заміною", filename)
    return file_bytes.decode("latin-1", errors="replace")

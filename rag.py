"""
rag.py — RAG-підсистема: ChromaDB, чанкінг, індексація, семантичний пошук.
"""

import re
import logging
import datetime
import requests

import chromadb

from config import (
    CHROMA_DIR, KNOWLEDGE_DIR,
    EMBED_MODEL, OLLAMA_URL,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K,
)
import state as _state
from state import get_indexed_states, get_doc_summaries, save_persistent_state
from ocr import extract_text

log = logging.getLogger("rag")

# ── ChromaDB ─────────────────────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"},
)
log.info("ChromaDB ініціалізовано. Чанків у колекції: %d", collection.count())


# ── Ембединги ────────────────────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    """Отримати вектор ембедингу від Ollama (синхронно)."""
    log.debug("Отримуємо ембединг для тексту (%d символів)", len(text))
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text[:4000]},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]
    except Exception as exc:
        log.error("Помилка ембедингу: %s", exc)
        raise RuntimeError(f"[Embedding] Помилка: {exc}. Перевір: ollama pull {EMBED_MODEL}")


# ── Чанкінг ──────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    """
    Розбити текст на чанки з урахуванням структури документа.
    Пріоритет: абзаци → речення → hard split.
    """
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + 2 <= CHUNK_SIZE:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
                current = current[-CHUNK_OVERLAP:].strip() if CHUNK_OVERLAP > 0 else ""

            if len(para) > CHUNK_SIZE:
                sentences = re.split(r"(?<=[.!?…])\s+", para)
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    if len(current) + len(sent) + 1 <= CHUNK_SIZE:
                        current = (current + " " + sent).strip()
                    else:
                        if current:
                            chunks.append(current)
                            current = current[-CHUNK_OVERLAP:].strip() if CHUNK_OVERLAP > 0 else ""
                        while len(sent) > CHUNK_SIZE:
                            chunks.append(sent[:CHUNK_SIZE])
                            sent = sent[CHUNK_SIZE - CHUNK_OVERLAP:]
                        current = sent
            else:
                current = (current + " " + para).strip()

    if current.strip():
        chunks.append(current.strip())

    result = [c for c in chunks if len(c.strip()) >= 30]
    log.debug("chunk_text: %d чанків згенеровано", len(result))
    return result


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
    preview = preview_raw[: last_dot + 1] if last_dot > 100 else preview_raw[:600]

    headings: list[str] = []
    for line in lines[:30]:
        stripped = line.strip()
        if (stripped.isupper() and 5 < len(stripped) < 80) or \
           (len(stripped) < 80 and stripped and stripped[0].isdigit()):
            headings.append(stripped[:80])
        if len(headings) >= 8:
            break

    summary = {
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
    log.debug("Резюме для '%s': %d символів, %d слів, мова=%s", filename, chars, len(words), lang)
    return summary


# ── Індексація ────────────────────────────────────────────────────────────────

def index_document(filename: str, text: str) -> int:
    """Проіндексувати документ: розбити на чанки → ембединги → ChromaDB."""
    log.info("Індексація документа: %s", filename)

    try:
        existing = collection.get(where={"source": filename})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            log.debug("Видалено старих чанків: %d для '%s'", len(existing["ids"]), filename)
    except Exception as exc:
        log.warning("Не вдалося видалити старі чанки для '%s': %s", filename, exc)

    chunks = chunk_text(text)
    if not chunks:
        log.warning("%s: порожній текст після чанкінгу, пропускаємо", filename)
        return 0

    log.info("Індексуємо '%s': %d чанків...", filename, len(chunks))

    ids, embeddings, documents, metadatas = [], [], [], []
    for i, chunk in enumerate(chunks):
        try:
            emb = get_embedding(chunk)
        except RuntimeError as exc:
            log.warning("Пропускаємо чанк %d через помилку ембедингу: %s", i, exc)
            continue
        ids.append(f"{filename}__chunk_{i}")
        embeddings.append(emb)
        documents.append(chunk)
        metadatas.append({
            "source":       filename,
            "chunk_index":  i,
            "total_chunks": len(chunks),
        })

    if ids:
        collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        log.info("✓ '%s' — %d чанків збережено у ChromaDB", filename, len(ids))

    get_doc_summaries()[filename] = build_doc_summary(filename, text, len(ids))
    return len(ids)


# ── Синхронізація бази знань ──────────────────────────────────────────────────

def sync_knowledge_base() -> None:
    """
    Синхронізація бази знань з ChromaDB:
    - нові файли → індексуємо
    - змінені файли → переіндексуємо
    - видалені файли → видаляємо з ChromaDB
    - незмінені файли → пропускаємо
    """
    log.info("Починаємо синхронізацію бази знань...")

    # Беремо живі посилання на словники через геттери
    indexed = get_indexed_states()
    summaries = get_doc_summaries()

    files     = sorted(KNOWLEDGE_DIR.glob("*"))
    supported = {".txt", ".md", ".pdf", ".docx"}
    docs      = [f for f in files if f.suffix.lower() in supported and f.name != "README.txt"]

    current_states = {f.name: f.stat().st_mtime for f in docs}
    log.debug("Знайдено файлів у knowledge_base: %d", len(docs))

    for f in docs:
        saved_mtime   = indexed.get(f.name)
        current_mtime = current_states[f.name]

        if saved_mtime == current_mtime:
            try:
                existing = collection.get(where={"source": f.name})
                if existing["ids"]:
                    log.debug("✓ '%s' вже в базі (%d чанків), пропускаємо", f.name, len(existing["ids"]))
                    continue
                else:
                    log.warning("'%s': є у стані, але відсутній у ChromaDB → переіндексуємо", f.name)
            except Exception as exc:
                log.warning("'%s': помилка перевірки ChromaDB → переіндексуємо. %s", f.name, exc)

        action = "нового" if saved_mtime is None else "оновленого"
        log.info("Індексація %s файлу: %s", action, f.name)
        try:
            text = extract_text(f.read_bytes(), f.name)
            if text.strip():
                n = index_document(f.name, text)
                if n > 0:
                    indexed[f.name] = current_mtime
                    save_persistent_state()
            else:
                log.warning("'%s': не вдалося витягти текст, пропускаємо", f.name)
        except Exception as exc:
            log.error("Помилка індексації '%s': %s", f.name, exc)

    deleted = set(indexed) - set(current_states)
    for name in deleted:
        try:
            existing = collection.get(where={"source": name})
            if existing["ids"]:
                collection.delete(ids=existing["ids"])
            del indexed[name]
            summaries.pop(name, None)
            save_persistent_state()
            log.info("Видалено з ChromaDB: %s", name)
        except Exception as exc:
            log.error("Помилка видалення '%s' з ChromaDB: %s", name, exc)

    total = collection.count()
    log.info("База знань синхронізована. Всього чанків у ChromaDB: %d", total)


# ── Пошук ─────────────────────────────────────────────────────────────────────

def retrieve_chunks(query: str, n: int = TOP_K) -> list[dict]:
    """
    Семантичний пошук: повертає топ-N найрелевантніших чанків.
    Результат: [{"text": ..., "source": ..., "score": ...}]
    """
    total = collection.count()
    log.debug("retrieve_chunks: запит='%s...', n=%d, всього чанків=%d", query[:60], n, total)

    if total == 0:
        log.warning("retrieve_chunks: ColecDB порожня, повертаємо []")
        return []

    n_results = min(n, total)
    try:
        emb = get_embedding(query)
        results = collection.query(
            query_embeddings=[emb],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = round(1.0 - dist, 3)
            chunks.append({
                "text":   doc,
                "source": meta.get("source", "невідомо"),
                "score":  similarity,
            })
        chunks.sort(key=lambda x: x["score"], reverse=True)
        log.debug("retrieve_chunks: знайдено %d чанків", len(chunks))
        return chunks
    except Exception as exc:
        log.error("Помилка RAG-пошуку: %s", exc)
        return []


def build_rag_context(
    seller_text: str, buyer_text: str, case_docs_text: str = ""
) -> tuple[str, list[dict]]:
    """
    Формуємо RAG-запит з тексту сторін та витягуємо релевантні чанки.
    Повертає: (контекст для промпту, список чанків для логування)
    """
    query  = f"Спір щодо договору: {seller_text[:600]} {buyer_text[:600]} {case_docs_text[:400]}"
    chunks = retrieve_chunks(query, n=TOP_K)

    if not chunks:
        log.info("build_rag_context: релевантних чанків не знайдено")
        return "", []

    context_parts = [
        f"[Джерело: {ch['source']} | Релевантність: {ch['score']}]\n{ch['text']}"
        for i, ch in enumerate(chunks, 1)
    ]
    context = "\n\n" + ("─" * 50 + "\n").join(context_parts)
    log.info("build_rag_context: сформовано контекст з %d чанків", len(chunks))
    return context, chunks
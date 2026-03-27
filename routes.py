import asyncio
import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

from config import BASE_DIR, KNOWLEDGE_DIR
from ocr import extract_text
from rag import collection, retrieve_chunks, sync_knowledge_base
from state import get_doc_summaries, get_indexed_states, save_persistent_state
from analyzer import analyze_with_ollama
from documents import build_doc_summary, collect_party_text, extract_case_doc, extract_text_from_path

log = logging.getLogger("routes")

router = APIRouter()

ai_queue_lock   = asyncio.Lock()
waiting_in_queue = 0


# ── Pydantic-моделі (для /analyze-case) ──────────────────────────────────────

class AiAnswer(BaseModel):
    answerId: str
    created: str
    text: str | None = None
    attachmentFileKeys: List[str] = []

class AiParty(BaseModel):
    name: str
    answers: List[AiAnswer]

class AiCaseDoc(BaseModel):
    title: str
    file_key: str

class AiContractDetails(BaseModel):
    number: str
    date: str
    deliveryStart: str
    deliveryEnd: str
    totalCost: float
    totalVolume: float
    deliveredVolume: float
    executionPercent: float
    priceChangePercent: float

class CaseAnalysisRequest(BaseModel):
    caseNumber:   str
    buyer:        AiParty
    seller:       AiParty
    caseDocuments: List[AiCaseDoc] = []
    contractDetails: AiContractDetails



# ── Службові маршрути ─────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def index():
    log.debug("GET /")
    return (BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")


@router.get("/health")
async def health():
    import httpx
    from config import OLLAMA_URL, MODEL, EMBED_MODEL

    log.debug("GET /health")
    kb_files = [
        {"name": f.name, "chunks": None}
        for f in sorted(KNOWLEDGE_DIR.glob("*"))
        if f.suffix.lower() in {".txt", ".md", ".pdf", ".docx"} and f.name != "README.txt"
    ]
    for entry in kb_files:
        try:
            res = collection.get(where={"source": entry["name"]})
            entry["chunks"] = len(res["ids"])
        except Exception:
            entry["chunks"] = 0

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
        log.info("Health check: Ollama підключена, моделей: %d", len(models))
        return {
            "status":             "ok",
            "ollama":             "connected",
            "models":             models,
            "active_model":       MODEL,
            "embed_model":        EMBED_MODEL,
            "knowledge_base":     kb_files,
            "chroma_total_chunks": collection.count(),
            "rag":                "enabled",
        }
    except Exception as exc:
        log.warning("Health check: Ollama недоступна. %s", exc)
        from config import EMBED_MODEL
        return {
            "status":       "ok",
            "ollama":       "disconnected",
            "error":        str(exc),
            "active_model": MODEL,
            "embed_model":  EMBED_MODEL,
            "knowledge_base": kb_files,
            "chroma_total_chunks": collection.count(),
            "rag": "enabled",
        }


# ── Knowledge Base маршрути ───────────────────────────────────────────────────

@router.get("/knowledge-base")
async def kb_list():
    log.debug("GET /knowledge-base")
    files = []
    for f in sorted(KNOWLEDGE_DIR.glob("*")):
        if f.suffix.lower() in {".txt", ".md", ".pdf", ".docx"} and f.name != "README.txt":
            try:
                res    = collection.get(where={"source": f.name})
                chunks = len(res["ids"])
            except Exception:
                chunks = 0
            files.append({
                "name":     f.name,
                "size_kb":  round(f.stat().st_size / 1024, 1),
                "chunks":   chunks,
                "indexed":  chunks > 0,
            })
    log.info("kb_list: %d файлів", len(files))
    return {"files": files, "count": len(files), "total_chunks": collection.count()}


@router.post("/knowledge-base/upload")
async def kb_upload(files: List[UploadFile] = File(...)):
    log.info("POST /knowledge-base/upload: %d файлів", len(files))
    saved = []
    for uf in files:
        if not uf.filename:
            continue
        ext = Path(uf.filename).suffix.lower()
        if ext not in {".txt", ".md", ".pdf", ".docx"}:
            log.warning("Непідтримуваний формат: %s", uf.filename)
            raise HTTPException(400, f"Непідтримуваний формат: {uf.filename}")
        dest = KNOWLEDGE_DIR / uf.filename
        dest.write_bytes(await uf.read())
        saved.append(uf.filename)
        log.info("Збережено файл: %s", uf.filename)

    sync_knowledge_base()
    log.info("Переіндексацію завершено після завантаження. Збережено: %s", saved)
    return {
        "saved":        saved,
        "message":      f"Збережено та проіндексовано {len(saved)} файлів",
        "total_chunks": collection.count(),
    }


@router.delete("/knowledge-base/{filename}")
async def kb_delete(filename: str):
    log.info("DELETE /knowledge-base/%s", filename)
    target = KNOWLEDGE_DIR / filename
    if not target.exists():
        raise HTTPException(404, "Файл не знайдено")
    target.unlink()
    sync_knowledge_base()
    log.info("Файл видалено: %s", filename)
    return {"deleted": filename, "total_chunks": collection.count()}


@router.post("/knowledge-base/reindex")
async def kb_reindex():
    """Примусова повна переіндексація всієї бази знань."""
    log.info("POST /knowledge-base/reindex — скидаємо стан і переіндексовуємо")
    get_indexed_states().clear()
    sync_knowledge_base()
    return {"message": "Переіндексацію завершено", "total_chunks": collection.count()}


@router.get("/knowledge-base/search")
async def kb_search(q: str, n: int = 5):
    """Тестовий ендпоінт для перевірки RAG-пошуку."""
    log.debug("GET /knowledge-base/search?q=%s&n=%d", q, n)
    if not q:
        raise HTTPException(400, "Параметр q є обов'язковим")
    chunks = retrieve_chunks(q, n=n)
    log.info("kb_search: знайдено %d чанків для запиту '%s'", len(chunks), q)
    return {"query": q, "results": chunks, "count": len(chunks)}


@router.get("/knowledge-base/summaries")
async def kb_summaries():
    """Повертає стислі огляди всіх проіндексованих документів."""
    log.debug("GET /knowledge-base/summaries")
    summaries = get_doc_summaries()
    return {"summaries": list(summaries.values()), "count": len(summaries)}


@router.get("/knowledge-base/summary/{filename}")
async def kb_summary_single(filename: str):
    """Огляд одного конкретного документа."""
    log.debug("GET /knowledge-base/summary/%s", filename)
    summaries = get_doc_summaries()
    if filename not in summaries:
        target = KNOWLEDGE_DIR / filename
        if not target.exists():
            raise HTTPException(404, f"Файл '{filename}' не знайдено")
        try:
            text    = extract_text(target.read_bytes(), filename)
            res     = collection.get(where={"source": filename})
            chunks  = len(res["ids"]) if res["ids"] else 0
            summary = build_doc_summary(filename, text, chunks)
            summaries[filename] = summary
            save_persistent_state()
            log.info("Згенеровано резюме для '%s'", filename)
            return summary
        except Exception as exc:
            log.error("Помилка генерації резюме для '%s': %s", filename, exc)
            raise HTTPException(500, str(exc))
    return summaries[filename]


# ── Аналіз через форму (мультипарт) ──────────────────────────────────────────

@router.post("/analyze")
async def analyze(
    seller_files:     List[UploadFile] = File(default=[]),
    buyer_files:      List[UploadFile] = File(default=[]),
    seller_text:      str = Form(""),
    buyer_text:       str = Form(""),
    contract_file:    UploadFile | None = File(default=None),
    schedule_file:    UploadFile | None = File(default=None),
    certificate_file: UploadFile | None = File(default=None),
):
    import httpx as _httpx
    import json as _json

    log.info(
        "POST /analyze: seller_files=%d, buyer_files=%d",
        len(seller_files), len(buyer_files),
    )

    final_seller, seller_metas = await collect_party_text(seller_files, seller_text, "Продавець")
    final_buyer,  buyer_metas  = await collect_party_text(buyer_files,  buyer_text,  "Покупець")

    if "не надала жодних пояснень" in final_seller and "не надала жодних пояснень" in final_buyer:
        raise HTTPException(400, "Необхідно надати дані хоча б для однієї зі сторін.")

    contract_text,    contract_meta    = await extract_case_doc(contract_file,    "ДОГОВІР КУПІВЛІ-ПРОДАЖУ")
    schedule_text,    schedule_meta    = await extract_case_doc(schedule_file,    "ГРАФІК ПОСТАВКИ")
    certificate_text, certificate_meta = await extract_case_doc(certificate_file, "БІРЖОВЕ СВІДОЦТВО")

    case_docs = {
        "contract":    contract_text,
        "schedule":    schedule_text,
        "certificate": certificate_text,
    }

    user_file_metas = [
        m for m in [contract_meta, schedule_meta, certificate_meta, *seller_metas, *buyer_metas]
        if m is not None
    ]

    try:
        result = await analyze_with_ollama(final_seller, final_buyer, case_docs)
        log.info("POST /analyze: аналіз завершено успішно")
        return JSONResponse({
            "success":         True,
            "result":          result,
            "user_docs":       user_file_metas,
            "has_contract":    bool(contract_text),
            "has_schedule":    bool(schedule_text),
            "has_certificate": bool(certificate_text),
        })
    except (_json.JSONDecodeError, ValueError) as exc:
        log.error("POST /analyze: неправильний формат відповіді моделі: %s", exc)
        raise HTTPException(500, f"Модель повернула неправильний формат: {exc}")
    except _httpx.ConnectError:
        log.error("POST /analyze: Ollama недоступна")
        raise HTTPException(503, "Ollama недоступна. Запустіть: ollama serve")
    except Exception as exc:
        log.error("POST /analyze: непередбачена помилка: %s", exc)
        raise HTTPException(500, str(exc))


# ── Аналіз через JSON (для Blazor) ───────────────────────────────────────────

@router.post("/analyze-case")
async def analyze_case(payload: CaseAnalysisRequest):

    global waiting_in_queue
    case_no = payload.caseNumber
    det = payload.contractDetails

    log.info(f"===> [ПОЧАТОК] Обробка справи № {case_no}")

    # --- ПЕРЕВІРКА НАЯВНОСТІ ТЕКСТОВИХ ПОЯСНЕНЬ ---
    # Збираємо всі тексти відповідей в один рядок для перевірки
    buyer_texts = [ans.text for ans in payload.buyer.answers if ans.text and ans.text.strip()]
    seller_texts = [ans.text for ans in payload.seller.answers if ans.text and ans.text.strip()]

    b_has_txt = "✅ Є ТЕКСТ" if buyer_texts else "❌ ПОРОЖНЬО"
    s_has_txt = "✅ Є ТЕКСТ" if seller_texts else "❌ ПОРОЖНЬО"

    log.info(f"[{case_no}] Статус пояснень сторін:")
    log.info(f"   - Покупець ({payload.buyer.name}): {b_has_txt} (к-ть відповідей: {len(buyer_texts)})")
    log.info(f"   - Продавець ({payload.seller.name}): {s_has_txt} (к-ть відповідей: {len(seller_texts)})")
    # ----------------------------------------------

    processed_docs: dict[str, str] = {}

    log.info(f"[{case_no}] Етап 1: Збір та зчитування документів...")

    async def get_text_safely(path_to_file, owner):
        try:
            raw = await extract_text_from_path(path_to_file)
            text = str(raw[0]) if isinstance(raw, tuple) else str(raw)
            if text.strip():
                log.info(f"   - [{case_no}] Файл прочитано: {Path(path_to_file).name} ({owner})")
                return text
        except Exception as e:
            log.error(f"   - [{case_no}] Помилка файлу {path_to_file}: {e}")
        return ""

    # Збираємо файли (Покупець)
    for ans in payload.buyer.answers:
        for f_path in ans.attachmentFileKeys:
            txt = await get_text_safely(f_path, "Покупець")
            if txt: processed_docs[f"Покупець_{Path(f_path).name}"] = txt

    # Збираємо файли (Продавець)
    for ans in payload.seller.answers:
        for f_path in ans.attachmentFileKeys:
            txt = await get_text_safely(f_path, "Продавець")
            if txt: processed_docs[f"Продавець_{Path(f_path).name}"] = txt

    # Документи справи
    for doc in payload.caseDocuments:
        txt = await get_text_safely(doc.file_key, "Справа")
        if txt: processed_docs[f"Справа_{doc.title}"] = txt

    log.info(f"[{case_no}] Всього зібрано файлів для аналізу: {len(processed_docs)}")

    log.info(f"[{case_no}] Етап 2: Очікування черги (Черга: {waiting_in_queue})...")

    try:
        waiting_in_queue += 1
        async with ai_queue_lock:
            waiting_in_queue -= 1
            log.info(f"[{case_no}] Початок генерації в Ollama (qwen3:14b)...")

            # Формуємо фінальні тексти для промпту
            full_buyer_text = "\n".join(buyer_texts) if buyer_texts else "Пояснення відсутні"
            full_seller_text = "\n".join(seller_texts) if seller_texts else "Пояснення відсутні"
            result = await analyze_with_ollama(
                seller_text=f"Назва: {payload.seller.name}\nПояснення: {full_seller_text}",
                buyer_text=f"Назва: {payload.buyer.name}\nПояснення: {full_buyer_text}",
                contract_details=payload.contractDetails,
                case_docs=processed_docs,
            )

        agent_conclusion = result.get("agent_conclusion", "").strip()
        log.info(f"[{case_no}] Етап 3: Аналіз успішний. Довжина відповіді: {len(agent_conclusion)} симв.")
        log.info(f"<=== [ЗАВЕРШЕНО] Справа № {case_no}")
        log.info("-" * 50)

        return PlainTextResponse(content=agent_conclusion)

    except Exception as exc:
        if waiting_in_queue > 0: waiting_in_queue -= 1
        log.error(f"!!! [КРИТИЧНА ПОМИЛКА] Справа № {case_no}: {exc}")
        return PlainTextResponse(content=f"Помилка аналізу: {exc}", status_code=500)
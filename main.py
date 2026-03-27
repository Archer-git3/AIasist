import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from config import STATIC_DIR, KNOWLEDGE_DIR, CHROMA_DIR
from logger import setup_logging
import state
from rag import sync_knowledge_base
from ocr import ocr_reader
from routes import router

# ── Ініціалізація логування ──────────────────────────────────────────────────
setup_logging()
log = logging.getLogger("main")

# ── Обов'язкові директорії (створюємо лише ті, без яких додаток не запуститься) ──
for d in (KNOWLEDGE_DIR, CHROMA_DIR):
    d.mkdir(exist_ok=True)
    log.debug("Директорія підтверджена: %s", d)


# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="Contract Analyzer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router)


@app.on_event("startup")
async def startup_event():
    log.info("Завантаження збереженого стану індексації...")
    state.load_persistent_state()
    log.info("Синхронізація бази знань з ChromaDB...")
    sync_knowledge_base()
    log.info("Додаток запущено та готовий до роботи.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
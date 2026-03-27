"""
logger.py — Налаштування системи логування для всього застосунку.
"""

import logging
import sys
from pathlib import Path


LOG_DIR = Path(__file__).parent / "logs"


def setup_logging(
    level: int = logging.DEBUG,
    log_to_file: bool = True,
) -> None:
    """
    Ініціалізує логування:
    - консоль (INFO і вище)
    - файл logs/app.log (DEBUG і вище, з ротацією)
    """
    LOG_DIR.mkdir(exist_ok=True)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # ── Консоль ──────────────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    # ── Файл ─────────────────────────────────────────────────────────────────
    if log_to_file:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            LOG_DIR / "app.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

    # Заглушаємо надмірно балакучі бібліотеки
    for noisy in ("httpx", "httpcore", "chromadb", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

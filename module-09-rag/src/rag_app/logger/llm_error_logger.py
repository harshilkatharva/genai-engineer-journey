import logging
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

llm_error_logger = logging.getLogger("llm_client_error")
llm_error_logger.setLevel(logging.INFO)

# Prevent duplicate handlers if imported multiple times
if not llm_error_logger.handlers:
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = logging.FileHandler(LOG_DIR / "llm_client_error.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    llm_error_logger.addHandler(file_handler)
    llm_error_logger.addHandler(console_handler)

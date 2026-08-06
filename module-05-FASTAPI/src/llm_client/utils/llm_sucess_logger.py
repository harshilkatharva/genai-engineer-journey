import logging
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

llm_sucess_logger = logging.getLogger("llm_client_sucess")
llm_sucess_logger.setLevel(logging.INFO)

# Prevent duplicate handlers if imported multiple times
if not llm_sucess_logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        LOG_DIR / "llm_client_sucess.log", encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    llm_sucess_logger.addHandler(file_handler)
    llm_sucess_logger.addHandler(console_handler)

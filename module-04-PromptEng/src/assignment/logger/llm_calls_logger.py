import logging
from pathlib import Path

LOG_DIR = Path("src/assignment/logs")
LOG_DIR.mkdir(exist_ok=True)

llm_call_logger = logging.getLogger("llm_calls")
llm_call_logger.setLevel(logging.INFO)

if not llm_call_logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(LOG_DIR / "llm_calls.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    llm_call_logger.addHandler(file_handler)
    llm_call_logger.addHandler(console_handler)

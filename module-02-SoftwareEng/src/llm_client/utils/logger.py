import logging
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("llm_client")
logger.setLevel("INFO")

file_handler = logging.FileHandler(LOG_DIR / "llm_client.log")

console_handler = logging.StreamHandler()

logger.addHandler(file_handler)
logger.addHandler(console_handler)

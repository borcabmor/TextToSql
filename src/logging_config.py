import logging

from src.utils import get_project_folder
from pathlib import Path


def setup_logging(nivel: str, log_filename: str = "app.log"):
    path_proyecto = get_project_folder()
    Path(path_proyecto / "logs").mkdir(exist_ok=True)
    fichero_log = path_proyecto / "logs" / log_filename

    logging.basicConfig(
        level=getattr(logging, nivel.upper(), logging.DEBUG),
        format="%(asctime)s | %(levelname)-8s | %(filename)-18s | %(funcName)-14s:%(lineno)-4s | %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(),  # Salida estandard
            logging.FileHandler(fichero_log),  # Salida a archivo
        ],
    )

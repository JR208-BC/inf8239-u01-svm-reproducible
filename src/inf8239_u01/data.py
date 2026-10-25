"""Descarga y carga del dataset (Predict Students' Dropout and Academic Success)."""
from pathlib import Path

import pandas as pd

UCI_URL = ("https://archive.ics.uci.edu/static/public/697/"
           "predict+students+dropout+and+academic+success.zip")
DEFAULT_PATH = "data/raw/dataset.csv"
EXPECTED_SHAPE = (4424, 37)


def download_csv(url: str, destination=DEFAULT_PATH, **read_kwargs) -> Path:
    """Descarga un CSV (o un .zip con un solo CSV) y lo guarda separado por comas."""
    if not url.startswith(("https://", "http://")):
        raise ValueError("La fuente debe ser una URL HTTP(S)")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(url, **read_kwargs)
    if frame.empty:
        raise ValueError("El dataset descargado está vacío")
    frame.to_csv(path, index=False)
    return path


def load_dataset(path=DEFAULT_PATH) -> pd.DataFrame:
    """Lee el CSV local y limpia espacios o tabuladores ocultos en los nombres de columna."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecute: python scripts/download_data.py")
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    return frame

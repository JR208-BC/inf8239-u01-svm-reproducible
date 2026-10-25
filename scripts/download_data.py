"""Descarga reproducible del dataset a data/raw/dataset.csv.

Uso (desde la raíz del repositorio):
    python scripts/download_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inf8239_u01.data import EXPECTED_SHAPE, UCI_URL, download_csv, load_dataset  # noqa: E402

if __name__ == "__main__":
    path = download_csv(UCI_URL, destination="data/raw/dataset.csv", sep=";")
    frame = load_dataset(path)
    print(f"Guardado en {path} con forma {frame.shape}")
    if frame.shape != EXPECTED_SHAPE:
        raise SystemExit(f"Forma inesperada {frame.shape}; se esperaba {EXPECTED_SHAPE}")
    print("Verificación correcta:", frame["Target"].value_counts().to_dict())

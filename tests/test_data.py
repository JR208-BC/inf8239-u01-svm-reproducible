import pandas as pd
import pytest

from inf8239_u01 import data


def test_download_rejects_non_http_source(tmp_path):
    with pytest.raises(ValueError):
        data.download_csv("C:/Users/Nombre/dataset.csv", destination=tmp_path / "x.csv")


def test_download_rejects_empty_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(data.pd, "read_csv", lambda *a, **k: pd.DataFrame())
    with pytest.raises(ValueError):
        data.download_csv("https://ejemplo.org/x.csv", destination=tmp_path / "x.csv")


def test_download_saves_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(data.pd, "read_csv", lambda *a, **k: pd.DataFrame({"a": [1, 2]}))
    path = data.download_csv("https://ejemplo.org/x.csv", destination=tmp_path / "x.csv")
    assert pd.read_csv(path).shape == (2, 1)


def test_load_dataset_strips_column_names(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"Daytime/evening attendance\t": [1]}).to_csv(p, index=False)
    assert list(data.load_dataset(p).columns) == ["Daytime/evening attendance"]

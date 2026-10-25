import pandas as pd
import pytest

from inf8239_u01.data import DEFAULT_PATH, load_dataset
from inf8239_u01.features import TARGET

pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path(DEFAULT_PATH).exists(),
    reason="Falta data/raw/dataset.csv (ejecute scripts/download_data.py)")


def test_dataset_is_not_empty():
    assert not load_dataset().empty


def test_required_columns_exist():
    assert {TARGET, "Course"} <= set(load_dataset().columns)


def test_target_has_no_missing_and_three_classes():
    y = load_dataset()[TARGET]
    assert y.notna().all()
    assert set(y.unique()) == {"Dropout", "Enrolled", "Graduate"}

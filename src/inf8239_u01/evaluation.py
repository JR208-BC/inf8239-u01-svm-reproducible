"""Métricas y tabla de errores."""
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             recall_score)

PRIORITY_CLASS = "Dropout"   # el error más costoso es no detectar un abandono


def summarize(y_true, y_pred) -> dict:
    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        "recall_dropout": recall_score(y_true, y_pred, labels=[PRIORITY_CLASS],
                                       average=None)[0],
    }


def per_class_table(y_true, y_pred) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    classes = [k for k in rep if k not in ("accuracy", "macro avg", "weighted avg")]
    return pd.DataFrame(rep).T.loc[classes, ["precision", "recall", "f1-score", "support"]]


def error_table(X, y_true, y_pred) -> pd.DataFrame:
    """Casos mal clasificados, con índice original, real y predicho."""
    wrong = pd.Series(y_true.values != pd.Series(y_pred).values, index=y_true.index)
    out = X[wrong.values].copy()
    out.insert(0, "real", y_true[wrong.values].values)
    out.insert(1, "predicho", pd.Series(y_pred, index=y_true.index)[wrong.values].values)
    return out

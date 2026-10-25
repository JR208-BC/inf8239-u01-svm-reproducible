from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import build_preprocessor


def build_svm(C=1.0, gamma="scale"):
    """SVM RBF para datos numéricos (ejemplo guiado de cáncer de mama)."""
    if C <= 0:
        raise ValueError("C debe ser positivo")
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", SVC(C=C, gamma=gamma, kernel="rbf", probability=True, random_state=42))
    ])


def build_dropout_baseline(X):
    """Baseline: predice siempre la clase mayoritaria."""
    return Pipeline([("prep", build_preprocessor(X)),
                     ("model", DummyClassifier(strategy="most_frequent"))])


def build_dropout_svm(X, C=1.0, gamma="scale", class_weight=None):
    """Pipeline SVM con preprocesamiento (el preprocesador se ajusta solo con train)."""
    if C <= 0:
        raise ValueError("C debe ser positivo")
    return Pipeline([("prep", build_preprocessor(X)),
                     ("model", SVC(C=C, gamma=gamma, kernel="rbf",
                                   class_weight=class_weight, random_state=42))])

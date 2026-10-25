from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import build_preprocessor, build_preprocessor_pca


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


def build_model_catalog(X):
    """Catálogo homogéneo del Ejercicio 02: seis configuraciones, mismo preprocesamiento.

    Todas comparten el preprocesador denso (necesario para HistGradientBoosting)
    y se ajustan sobre la misma partición, para que la comparación sea justa.
    No se balancean clases aquí: el objetivo de este ejercicio es comparar
    familias de modelos y costo, no repetir la elección del Ejercicio 01.
    """
    def pipe(model):
        return Pipeline([("prep", build_preprocessor(X, dense=True)), ("model", model)])

    return {
        "logistic": pipe(LogisticRegression(max_iter=2000)),
        "svm_c1": pipe(SVC(C=1, probability=True, random_state=42)),
        "svm_c10": pipe(SVC(C=10, probability=True, random_state=42)),
        "rf_100": pipe(RandomForestClassifier(n_estimators=100, max_depth=8,
                                              random_state=42, n_jobs=-1)),
        "rf_300": pipe(RandomForestClassifier(n_estimators=300, max_depth=8,
                                              random_state=42, n_jobs=-1)),
        "boost": pipe(HistGradientBoostingClassifier(max_depth=6, random_state=42)),
    }


def build_dropout_svm_pca(X, C=1.0, n_components=0.95):
    """Variante de reducción: SVM sobre PCA del bloque numérico + one-hot de categóricas."""
    if C <= 0:
        raise ValueError("C debe ser positivo")
    return Pipeline([("prep", build_preprocessor_pca(X, n_components=n_components)),
                     ("model", SVC(C=C, probability=True, random_state=42))])

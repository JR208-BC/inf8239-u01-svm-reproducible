"""Definición de variables, prevención de fuga y preprocesamiento."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Target"

# Momento de predicción: fin del 1.er semestre.
# Se excluye lo que ocurre después (2.º semestre) o lo que puede registrarse
# cuando el estudiante ya está en riesgo (deuda y pago de matrícula).
DROP_COLUMNS = [
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Debtor",
    "Tuition fees up to date",
]

# Códigos de categoría (no son cantidades): van a one-hot.
CATEGORICAL_COLUMNS = [
    "Marital status", "Application mode", "Course", "Previous qualification",
    "Nacionality", "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
]


def split_features_target(frame: pd.DataFrame):
    """Devuelve (X, y) sin el target ni las columnas con riesgo de fuga."""
    if TARGET not in frame.columns:
        raise KeyError(f"Falta la columna objetivo '{TARGET}'")
    if frame[TARGET].isna().any():
        raise ValueError("El target tiene valores ausentes")
    X = frame.drop(columns=[TARGET] + [c for c in DROP_COLUMNS if c in frame.columns])
    return X, frame[TARGET]


def build_preprocessor(X: pd.DataFrame, dense: bool = False) -> ColumnTransformer:
    """Imputa, escala las numéricas y codifica las categóricas.

    Este dataset no tiene ausentes (verificado en la auditoría), pero se
    incluye el imputer para que el pipeline sea robusto a datos futuros con
    huecos. Todo se ajusta dentro del Pipeline, así que solo ve entrenamiento.

    ``dense=True`` fuerza una salida densa (necesaria para
    HistGradientBoostingClassifier y para PCA/t-SNE, que no aceptan matrices
    dispersas).
    """
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                  sparse_output=not dense))])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


def build_preprocessor_pca(X: pd.DataFrame, n_components=0.95) -> ColumnTransformer:
    """Variante de reducción: PCA solo sobre el bloque numérico.

    Las categóricas se codifican con one-hot tal cual (sin reducir), porque
    aplicar PCA sobre columnas one-hot mezcla variables sin relación numérica
    real y no sería interpretable. Salida densa, requerida por PCA.
    """
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()),
                         ("pca", PCA(n_components=n_components, random_state=42))])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                  sparse_output=False))])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

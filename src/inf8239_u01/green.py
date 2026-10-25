"""Frontera de Pareto costo-beneficio (Green AI): rendimiento frente a costo."""
import pandas as pd


def pareto_flags(df: pd.DataFrame, score: str = "f1_macro", cost: str = "fit_median_s"):
    """True para cada fila que ningún otro modelo domina.

    Un modelo A domina a B si A tiene score igual o mejor Y costo igual o
    menor que B, con al menos una desigualdad estricta. Un modelo no
    dominado por ninguno está en la frontera de Pareto.
    """
    flags = []
    for _, row in df.iterrows():
        dominated = ((df[score] >= row[score]) & (df[cost] <= row[cost]) &
                    ((df[score] > row[score]) | (df[cost] < row[cost]))).any()
        flags.append(not bool(dominated))
    return flags

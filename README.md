# INF-8239 · Ejercicios 01 y 02 · Dataset público, SVM, ensambles y Green AI

¿Puede un modelo identificar, con datos disponibles al terminar el 1.er semestre, si un estudiante abandonará (`Dropout`), seguirá matriculado (`Enrolled`) o se graduará (`Graduate`)?

- **Dataset:** [Predict Students' Dropout and Academic Success](https://doi.org/10.24432/C5MC89) (UCI, id 697, CC BY 4.0). 4,424 estudiantes, 36 variables. Mismo dataset, target, partición y exclusiones por fuga en ambos ejercicios.
- **Ejercicio 01** (SVM reproducible): F1 macro en prueba de 0.222 (baseline) → 0.598 (SVM) → 0.640 (SVM balanceado). Detalle en `reports/metrics.csv` y `docs/CONCLUSION.md`.
- **Ejercicio 02** (ensambles, reducción y Green AI): siete configuraciones comparadas con medición repetida; máximo F1 macro 0.640 (boosting); modelo elegido por costo/beneficio: regresión logística (F1 0.631, 19.7 % más rápida de entrenar, 98.3 % más liviana). Detalle en `reports/green_ai_results.csv` y `docs/DECISION_GREEN_AI.md`.
- **Aviso:** ejercicio académico; no usar para decidir sobre estudiantes reales.

## Instalación

Requiere Python 3.11 o superior. Desde la raíz del repositorio:

```bash
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Linux / macOS:         source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Versiones con las que se ejecutó: ver `reports/environment.txt` (Ejercicio 01) y `reports/environment_ejercicio02.txt` (Ejercicio 02).

## Descarga reproducible de los datos

```bash
python scripts/download_data.py
```

Descarga el `.zip` oficial del repositorio UCI, guarda `data/raw/dataset.csv` (separado por comas) y verifica que la forma sea (4424, 37). Los datos no se versionan (`data/raw/*` está en `.gitignore`).

## Ejecutar

```bash
python -m pytest -q                       # pruebas (29)
jupyter lab notebooks/
```

Para reejecutar los notebooks completos sin abrir Jupyter:

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 03_ejercicio01_svm_reproducible.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_ejercicio02_ensambles_green_ai.ipynb
```

- `03_ejercicio01_svm_reproducible.ipynb`: descarga los datos si faltan, hace la auditoría, aplica la prevención de fuga, entrena baseline y SVM, valida con validación cruzada, evalúa una sola vez en prueba y escribe sus resultados en `reports/`.
- `04_ejercicio02_ensambles_green_ai.ipynb`: reutiliza esa misma partición; compara regresión logística, dos SVM, dos Random Forest, boosting y una variante con PCA, con tres repeticiones de tiempo y mediana; genera dos mapas t-SNE, la frontera de Pareto y la decisión de costo/beneficio. También escribe en `reports/`.

Los modelos serializados (`reports/models/*.joblib`) se regeneran al ejecutar los notebooks; no se versionan (`.gitignore`).

## Estructura

```
src/inf8239_u01/   código reutilizable (data, features, models, evaluation, green)
tests/             pruebas con pytest (29)
scripts/           descarga reproducible
notebooks/         00 verificación · 01 SVM guiada · 02 exploración · 03 Ejercicio 01 · 04 Ejercicio 02
docs/              fichas de dataset, diccionario de datos, conclusión del Ej. 01, decisión Green AI del Ej. 02
reports/           métricas, figuras, errores, entorno, modelos serializados
```

## Decisiones clave

- **Partición:** 80/20 estratificada, `random_state=42`, igual en ambos ejercicios. El conjunto de prueba se usa una sola vez por modelo.
- **Fuga:** se excluyen las variables del 2.º semestre, `Debtor` y `Tuition fees up to date` (ver `reports/columnas_excluidas.csv`).
- **Preprocesamiento dentro del `Pipeline`:** el escalado, la imputación y la codificación solo aprenden de entrenamiento.
- **Métrica principal:** F1 macro; recall de `Dropout` como criterio de desempate (el error más costoso es no detectar un abandono).
- **Ejercicio 02 — reducción:** PCA solo sobre el bloque numérico (19 → 16 componentes al 95 % de varianza); las categóricas no se reducen, porque son la mayoría de las columnas y aplicar PCA a un one-hot mezclaría variables sin relación numérica real.
- **Ejercicio 02 — decisión:** se eligió la regresión logística por estar en la frontera de Pareto con la mejor relación F1/costo, no el modelo de mayor F1 (boosting). Justificación completa en `docs/DECISION_GREEN_AI.md`.

## Licencia y cita

Datos bajo CC BY 4.0. Cita: Realinho, V., Martins, M. V., Machado, J. y Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89

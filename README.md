# INF-8239 · Ejercicio 01 · Dataset público y SVM reproducible

¿Puede un SVM identificar, con datos disponibles al terminar el 1.er semestre, si un estudiante abandonará (`Dropout`), seguirá matriculado (`Enrolled`) o se graduará (`Graduate`)?

- **Dataset:** [Predict Students' Dropout and Academic Success](https://doi.org/10.24432/C5MC89) (UCI, id 697, CC BY 4.0). 4,424 estudiantes, 36 variables.
- **Resultado principal:** F1 macro en prueba de 0.222 (baseline) → 0.598 (SVM) → 0.640 (SVM balanceado). Detalle en `reports/metrics.csv` y en la conclusión (`docs/CONCLUSION.md`).
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

Versiones con las que se ejecutó: ver `reports/environment.txt`.

## Descarga reproducible de los datos

```bash
python scripts/download_data.py
```

Descarga el `.zip` oficial del repositorio UCI, guarda `data/raw/dataset.csv` (separado por comas) y verifica que la forma sea (4424, 37). Los datos no se versionan (`data/raw/*` está en `.gitignore`).

## Ejecutar

```bash
python -m pytest -q                       # pruebas
jupyter lab notebooks/03_ejercicio01_svm_reproducible.ipynb
```

Para reejecutar el notebook completo sin abrir Jupyter:

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 03_ejercicio01_svm_reproducible.ipynb
```

El notebook descarga los datos si faltan, hace la auditoría, aplica la prevención de fuga, entrena baseline y SVM, valida con validación cruzada, evalúa una sola vez en prueba y escribe sus resultados en `reports/`.

## Estructura

```
src/inf8239_u01/   código reutilizable (data, features, models, evaluation)
tests/             pruebas con pytest
scripts/           descarga reproducible
notebooks/         00 verificación · 01 SVM guiada · 02 exploración · 03 entrega del ejercicio
docs/              ficha comparativa, diccionario de datos, conclusión
reports/           métricas, figuras, errores, entorno
```

## Decisiones clave

- **Partición:** 80/20 estratificada, `random_state=42`. El conjunto de prueba se usa una sola vez.
- **Fuga:** se excluyen las variables del 2.º semestre, `Debtor` y `Tuition fees up to date` (ver `reports/columnas_excluidas.csv`).
- **Preprocesamiento dentro del `Pipeline`:** el escalado y la codificación solo aprenden de entrenamiento.
- **Métrica principal:** F1 macro; recall de `Dropout` como criterio de desempate.

## Licencia y cita

Datos bajo CC BY 4.0. Cita: Realinho, V., Martins, M. V., Machado, J. y Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89

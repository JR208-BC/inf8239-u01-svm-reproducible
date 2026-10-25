# Comparación de candidatos (LAB02, paso 2-3)

**Formulario previo:** ver `docs/ficha_dataset.md`.

| Criterio | Candidato A | Candidato B |
|---|---|---|
| Nombre | Predict Students' Dropout and Academic Success | Breast Cancer Wisconsin (Diagnostic) |
| Procedencia | UCI Machine Learning Repository, id 697. Realinho, Martins, Machado y Baptista, Instituto Politécnico de Portalegre. Ficha: https://archive.ics.uci.edu/dataset/697 | UCI Machine Learning Repository, id 17. Wolberg, Mangasarian y Street. Ficha: https://archive.ics.uci.edu/dataset/17 |
| Licencia | CC BY 4.0 (DOI 10.24432/C5MC89) | CC BY 4.0 (DOI 10.24432/C5DW2B) |
| Filas/columnas | 4,424 filas, 36 columnas + target | 569 filas, 30 columnas + target |
| Target y clases | `Target`: Dropout (32.1 %), Enrolled (17.9 %), Graduate (49.9 %) — 3 clases | `diagnosis`: maligno (37.3 %), benigno (62.7 %) — 2 clases |
| Ausentes | 0 (verificado) | 0 (verificado) |
| Riesgo de fuga | Real: 6 variables del 2.º semestre no existen al momento de predecir; `Debtor` y `Tuition fees up to date` pueden registrarse cuando el estudiante ya está en riesgo | Ninguno: todas las variables se miden de la misma imagen diagnóstica, antes del resultado |

## Criterios de aceptación (LAB02, paso 3)

| Criterio | Candidato A | Candidato B |
|---|---|---|
| Al menos 500 filas | Cumple (4,424) | Cumple (569) |
| Target observable, mínimo dos clases | Cumple (3 clases) | Cumple (2 clases) |
| Datos permitidos para uso académico | Cumple (CC BY 4.0) | Cumple (CC BY 4.0) |
| Variables disponibles al momento de predecir | Cumple, tras excluir las 8 columnas con riesgo de fuga | Cumple, sin exclusiones |
| Tamaño compatible con CPU/Colab gratuito | Cumple | Cumple |
| No repetido con otro estudiante sin autorización | Pendiente de confirmar con el docente | Pendiente de confirmar con el docente |

## Decisión

**Candidato aprobado: A — Predict Students' Dropout and Academic Success.**

Motivo: responde directamente la decisión del formulario (identificar riesgo de abandono), obliga a resolver un riesgo de fuga real (a diferencia de B, que no lo tiene) y deja una clase minoritaria (`Enrolled`) con error sustantivo que analizar. B se usa únicamente como ejercicio guiado del LAB01 (`notebooks/01_svm_guiada.ipynb`), no como candidato final.

**Nota de proceso.** Según el LAB02, esta comparación debe enviarse al docente antes de continuar con el entrenamiento final; el entrenamiento de este repositorio se hizo asumiendo esa aprobación. Confirme con el profesor antes de entregar.

Aprobado por: [nombre del estudiante] · Fecha: [fecha]

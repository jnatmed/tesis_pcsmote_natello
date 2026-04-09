# PC-SMOTE: Una variante de SMOTE controlada por percentiles

Este repositorio contiene la implementación y los experimentos asociados a la tesina de grado **"PC-SMOTE: Una variante de SMOTE controlada por percentiles"** (2026), presentada por **Juan Natello** en la Universidad Nacional de Luján.

El proyecto estudia el desbalance de clases en aprendizaje automático supervisado mediante una heurística geométrica basada en percentiles, orientada a regular la selección de semillas y la generación de muestras sintéticas según la estructura local de cada dataset.

## Resumen del Proyecto

PC-SMOTE (Percentile-Controlled SMOTE) es una variante de SMOTE que calibra radios de densidad y riesgo a partir de percentiles de distancias observadas, filtra semillas candidatas mediante densidad, riesgo y pureza local, y solo entonces interpola nuevas muestras sintéticas. En los experimentos de la tesis, este enfoque mostró mejoras en escenarios con separabilidad local suficiente y un comportamiento más conservador cuando la vecindad no resultó favorable para la interpolación.

En algunos datasets, una tasa alta de rechazo de semillas también sugirió que la conveniencia de sobremuestrear depende de la geometría local observada. Por eso, la narrativa del repositorio presenta ese comportamiento en clave exploratoria y apoyada en los resultados experimentales de la tesis.

### Fases del Algoritmo PC-SMOTE

1. **Fase 1: Definición heurística de radios mediante percentiles**: estima radios de densidad y riesgo a partir de la distribución empírica de distancias entre instancias minoritarias y sus vecinos más cercanos.
2. **Fase 2: Filtros geométricos de semillas**: evalúa cada semilla candidata según densidad, riesgo y pureza local (por proporción o entropía) antes de autorizar la interpolación.
3. **Fase 3: Generación adaptativa**: genera ejemplos sintéticos solo a partir de semillas aprobadas y vecinos de la misma clase ubicados dentro de una región considerada segura.

---

## Estructura del Repositorio

El repositorio está organizado siguiendo el pipeline de experimentación descrito en la tesis:

- [**`datasets/`**](datasets/README.md): gestión de fuentes de datos, carga y análisis exploratorio inicial.
- [**`scripts/`**](scripts/README.md): implementación central del algoritmo PC-SMOTE, limpieza opcional con Isolation Forest y herramientas de evaluación.
- [**`notebooks/`**](notebooks/README.md): experimentación, búsqueda de hiperparámetros y validación comparativa.
- [**`resultados/`**](resultados/README.md): logs detallados, tablas finales de métricas y mejores parámetros obtenidos.

---

## Metodología de Validación

Los experimentos siguen un pipeline estandarizado:

1. División de datos en train/test.
2. Escalado mediante `RobustScaler`.
3. Limpieza opcional con `Isolation Forest`, según la configuración evaluada.
4. Sobremuestreo comparativo (Original vs SMOTE vs Borderline-SMOTE vs ADASYN vs PC-SMOTE).
5. Entrenamiento de un clasificador `Random Forest`.
6. Evaluación mediante la métrica **F1-macro average**, junto con otras métricas de apoyo para interpretar el comportamiento de cada técnica.

## Referencias

Para una comprensión profunda de los fundamentos teóricos y matemáticos, consulte el archivo:

- `tesis_natello_pcsmote.pdf`: documento completo con el marco teórico, la formalización algorítmica y la discusión de resultados experimentales.

---

Desarrollado como parte de la Tesina de Grado para la Licenciatura en Sistemas de Información - UNLu (2026).

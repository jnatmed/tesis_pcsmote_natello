# PC-SMOTE: Una variante de SMOTE controlada por percentiles 

Este repositorio contiene la implementación y los experimentos asociados a la tesina de grado **"PC-SMOTE: Una variante de SMOTE controlada por percentiles"** (2026), presentada por **Juan Natello** en la Universidad Nacional de Luján.

El proyecto aborda el desafío del desbalance de clases en el aprendizaje automático supervisado, introduciendo un algoritmo de **doble acción** fundamentado en la autocalibración geométrica.

## 📝 Resumen del Proyecto

PC-SMOTE (Percentile-Controlled SMOTE) se distingue de los métodos tradicionales al actuar no solo como un generador dinámico de datos sintéticos en zonas seguras, sino también como un **escudo protector topológico** (sensor de inviabilidad). El algoritmo evalúa la morfología de cada clase mediante percentiles y filtros de pureza sustentados en entropía, frenando la interpolación en escenarios de solapamiento extremo para evitar la inyección de ruido perjudicial.

### 🛠️ Fases del Algoritmo PC-SMOTE

1.  **Fase 1: Auto-calibración Geométrica**: Utiliza percentiles dinámicos para definir radios de vecindad adaptativos (densidad y riesgo), ajustándose a la morfología específica de cada conjunto de datos.
2.  **Fase 2: Filtros Topológicos**: Evalúa cada punto de la clase minoritaria mediante filtros de densidad, riesgo y pureza (basada en entropía de Shannon). Las muestras que superan estos filtros se consideran "semillas válidas".
3.  **Fase 3: Generación Adaptativa**: Realiza la interpolación lineal entre semillas válidas y sus vecinos seguros, controlando el factor de sobregeneralización para mantener la integridad de las fronteras de decisión.

---

## 📂 Estructura del Repositorio

El repositorio está organizado siguiendo el pipeline de experimentación descrito en la tesis:

-   📁 [**`datasets/`**](datasets/README.md): Gestión de fuentes de datos, carga y análisis exploratorio inicial.
-   📁 [**`scripts/`**](scripts/README.md): Implementación central del algoritmo PC-SMOTE, limpieza con Isolation Forest y herramientas de evaluación.
-   📁 [**`notebooks/`**](notebooks/README.md): Experimentación, búsqueda de hiperparámetros y validación comparativa.
-   📁 [**`resultados/`**](resultados/README.md): Logs detallados, tablas de métricas final y mejores parámetros obtenidos.

---

## 🔬 Metodología de Validación

Los experimentos siguen un pipeline estandarizado:
1.  División de datos en Train/Test.
2.  Escalado mediante `RobustScaler`.
3.  Detección y limpieza de ruido con `Isolation Forest`.
4.  Sobremuestreo comparativo (Original vs SMOTE vs Borderline-SMOTE vs ADASYN vs PC-SMOTE).
5.  Entrenamiento de un clasificador *Random Forest*.
6.  Evaluación mediante la métrica **F1-macro average** para garantizar una medición equitativa ante el desbalance severo.

## 📄 Referencias

Para una comprensión profunda de los fundamentos teóricos y matemáticos, consulte el archivo:
-   `tesis_natello_pcsmote.pdf`: Documento completo con el marco teórico, formalización algorítmica y discusión de resultados experimentales.

---
*Desarrollado como parte de la Tesina de Grado para la Licenciatura en Sistemas de Información - UNLu (2026).*

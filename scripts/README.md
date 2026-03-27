[⬅️ Previo: Datasets](../datasets/README.md) | [Siguiente: Notebooks ➔](../notebooks/README.md) | [🏠 Inicio](../README.md)

# 📁 Scripts del Proyecto PC-SMOTE

Esta carpeta contiene todos los módulos de Python necesarios para la implementación del algoritmo PC-SMOTE, así como las herramientas auxiliares para el procesamiento de datos y la evaluación de resultados de acuerdo con la metodología de la tesina.

## 🛠️ Módulos Principales

- **`pc_smote.py`**: Implementación detallada del algoritmo **PC-SMOTE**. Incluye las tres fases clave: autocalibración por percentiles, filtrado topográfico (entropía/proporción) e interpolación adaptativa.
- **`isolation_cleaner.py`**: Utiliza el algoritmo *Isolation Forest* para la detección y limpieza de outliers en el conjunto de datos de entrenamiento, mejorando la robustez de las fronteras de decisión.
- **`evaluacion.py`**: Contiene las funciones para calcular métricas de rendimiento orientadas a datos desbalanceados, principalmente el **F1-macro average**, Precision y Recall.
- **`graficador_resultados.py` y `graficador2d.py`**: Herramientas para visualizar la distribución de las clases en 2D y el impacto del sobremuestreo en la estructura de los datos.
- **`meta_pcsmote.py`**: Módulo que facilita la configuración de hiperparámetros y la ejecución de PC-SMOTE dentro de pipelines de *scikit-learn*.
- **`Utils.py`**: Funciones auxiliares para la carga de datos, transformación de tipos y otras utilidades comunes.
- **`gestor_cache.py` y `cache.py`**: Lógica de almacenamiento en caché para optimizar los tiempos de ejecución durante experimentaciones repetitivas.
- **`limpiador.py`**: Scripts complementarios para la normalización y preprocesamiento de los datasets.

---
*Para más detalles sobre la implementación matemática de cada módulo, consulte el capítulo 3 (Metodología) y el Apéndice de la tesis.*

---
[⬅️ Previo: Datasets](../datasets/README.md) | [Siguiente: Notebooks ➔](../notebooks/README.md) | [🏠 Inicio](../README.md)

[⬅️ Previo: Datasets](../datasets/README.md) | [Siguiente: Notebooks ➜](../notebooks/README.md) | [🏠 Inicio](../README.md)

# Scripts del Proyecto PC-SMOTE

Esta carpeta contiene los módulos de Python necesarios para la implementación del algoritmo PC-SMOTE, así como utilidades auxiliares para procesamiento de datos y evaluación de resultados según la metodología de la tesina.

## Módulos Principales

- **`pc_smote.py`**: implementación detallada del algoritmo **PC-SMOTE**. Incluye las tres fases clave: definición heurística de radios por percentiles, filtrado geométrico de semillas (densidad, riesgo y pureza) e interpolación adaptativa.
- **`isolation_cleaner.py`**: limpieza opcional de outliers con `Isolation Forest` usando umbrales por percentil, en línea con el protocolo experimental de la tesis.
- **`evaluacion.py`**: funciones para calcular métricas de rendimiento orientadas a datos desbalanceados, principalmente **F1-macro average**, precisión y recall.
- **`graficador_resultados.py`** y **`graficador2d.py`**: herramientas para visualizar la distribución de clases en 2D y el impacto del sobremuestreo sobre la estructura local de los datos.
- **`meta_pcsmote.py`**: módulo que facilita la configuración de hiperparámetros y la ejecución de PC-SMOTE dentro de pipelines de `scikit-learn`.
- **`Utils.py`**: funciones auxiliares para carga de datos, transformación de tipos y utilidades comunes.
- **`gestor_cache.py`** y **`cache.py`**: lógica de almacenamiento en caché para optimizar tiempos de ejecución durante experimentaciones repetitivas.
- **`limpiador.py`**: scripts complementarios para normalización y preprocesamiento de datasets.

---

Para más detalles sobre la implementación matemática de cada módulo, consulte el capítulo 3 (Metodología) y el apéndice de código de la tesis.

---

[⬅️ Previo: Datasets](../datasets/README.md) | [Siguiente: Notebooks ➜](../notebooks/README.md) | [🏠 Inicio](../README.md)

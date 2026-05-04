[⬅️ Previo: Datasets](../datasets/README.md) | [Siguiente: Notebooks ➜](../notebooks/README.md) | [🏠 Inicio](../README.md)

# Scripts del Proyecto PC-SMOTE

Esta carpeta contiene los módulos de Python necesarios para la implementación del algoritmo PC-SMOTE, así como utilidades auxiliares para procesamiento de datos y evaluación de resultados según la metodología de la tesina.

## Módulos Principales

- **`pc_smote.py`**: implementación detallada del algoritmo **PC-SMOTE**. Incluye las tres fases clave: definición heurística de radios por percentiles, filtrado geométrico de semillas (densidad, riesgo y pureza) e interpolación adaptativa.
- **`contemporaneos/`**: implementación modular de técnicas contemporáneas de sobremuestreo con una base compartida `SamplerContemporaneo` para soporte binario y multiclase vía `one-vs-rest`. Actualmente incluye `RadiusSMOTE`, `LDSMOTE` y `VSSMOTE`.
- **`isolation_cleaner.py`**: limpieza opcional de outliers con `Isolation Forest` usando umbrales por percentil, en línea con el protocolo experimental de la tesis.
- **`evaluacion.py`**: funciones para calcular métricas de rendimiento orientadas a datos desbalanceados, principalmente **F1-macro average**, precisión y recall.
- **`graficador_resultados.py`** y **`graficador2d.py`**: herramientas para visualizar la distribución de clases en 2D y el impacto del sobremuestreo sobre la estructura local de los datos.
- **`meta_pcsmote.py`**: módulo que facilita la configuración de hiperparámetros y la ejecución de PC-SMOTE dentro de pipelines de `scikit-learn`.
- **`Utils.py`**: funciones auxiliares para carga de datos, transformación de tipos y utilidades comunes.
- **`gestor_cache.py`** y **`cache.py`**: lógica de almacenamiento en caché para optimizar tiempos de ejecución durante experimentaciones repetitivas.
- **`limpiador.py`**: scripts complementarios para normalización y preprocesamiento de datasets.
- **`generar_contemporaneos.py`**: toma los CSV base de `datasets/datasets_aumentados/base/` y genera nuevos datasets balanceados en `datasets/datasets_aumentados/contemporaneos/` con nombres compatibles con el pipeline actual.

---

## Relación con los notebooks

Los scripts se integran con los notebooks de evaluación bajo el criterio de **parámetros fijos**. En particular, `02_experimento_params_fijos.ipynb` utiliza las versiones base, clásicas y PC-SMOTE para entrenar `Random Forest` con una configuración constante. Por su parte, `03_experimento_modelos_alternativos_params_fijos.ipynb` permite extender la comparación a otros modelos y a datasets generados mediante las técnicas contemporáneas implementadas en `scripts/contemporaneos/`.

---

Para más detalles sobre la implementación matemática de cada módulo, consulte el capítulo 3 (Metodología) y el apéndice de código de la tesis.

---

[⬅️ Previo: Datasets](../datasets/README.md) | [Siguiente: Notebooks ➜](../notebooks/README.md) | [🏠 Inicio](../README.md)

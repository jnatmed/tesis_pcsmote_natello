[🏠 Volver al Inicio](../README.md) | [Siguiente: Scripts ➜](../scripts/README.md)

# Conjuntos de Datos (Datasets)

Este directorio centraliza las fuentes de datos utilizadas para la validación del algoritmo PC-SMOTE, así como los scripts y notebooks necesarios para su carga y análisis preliminar.

## Estructura de Datasets

- **`statlog+shuttle`**: conjunto de datos de alta disparidad para la detección de errores en transbordadores espaciales.
- **`US Crime`**: dataset con asimetría severa basado en registros criminales de EE. UU.
- **`telco_costumer_churn`**: datos de abandono de servicios de telecomunicación.
- **`predict_faults`**: escenario de mantenimiento predictivo industrial con fuerte solapamiento local, útil para analizar cuándo la heurística de PC-SMOTE restringe la interpolación.
- **`datasets_aumentados/`**: versiones de los conjuntos de datos tras la aplicación de técnicas de sobremuestreo para entrenamiento y comparación experimental.

## Herramientas de Carga y Configuración

- **`config_datasets.py`**: configuración centralizada para definir tipos de datos, rutas de archivos y esquemas para cada dataset.
- **`cargar_dataset.py`**: script automatizado para la importación estandarizada de todos los conjuntos evaluados en la tesis.
- **`analisis_datasets.ipynb`**: análisis exploratorio detallado de distribución de clases, solapamiento espacial y estructura geométrica inicial.

---

**Nota sobre reproducibilidad**: los datos crudos dentro de las carpetas de origen no deben modificarse manualmente para asegurar la integridad de los experimentos comparados en la tesis.

---

[🏠 Volver al Inicio](../README.md) | [Siguiente: Scripts ➜](../scripts/README.md)

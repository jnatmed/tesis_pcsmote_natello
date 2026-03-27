[🏠 Volver al Inicio](../README.md) | [Siguiente: Scripts ➔](../scripts/README.md)

# 📊 Conjuntos de Datos (Datasets)

Este directorio centraliza las fuentes de datos utilizadas para la validación del algoritmo PC-SMOTE, así como los scripts y notebooks necesarios para su carga y análisis preliminar.

## 📁 Estructura de Datasets

- **`statlog+shuttle`**: Conjunto de datos de alta disparidad para la detección de errores en transbordadores espaciales.
- **`US Crime`**: Dataset con asimetría severa basado en registros criminales de EE. UU.
- **`telco_costumer_churn`**: Datos de abandono de servicios de telecomunicación.
- **`predict_faults`**: Conjunto de datos de mantenimiento predictivo industrial, clave para probar el "sensor de inviabilidad" topográfica.
- **`datasets_aumentados/`**: Almacena versiones de los conjuntos de datos tras la aplicación de técnicas de sobremuestreo para su posterior uso en entrenamiento acelerado.

## 🛠️ Herramientas de Carga y Configuración

- **`config_datasets.py`**: Configuración centralizada para definir tipos de datos, rutas de archivos y esquemas para cada dataset.
- **`cargar_dataset.py`**: Script automatizado para la importación estandarizada de todos los conjuntos evaluados en la tesis.
- **`analisis_datasets.ipynb`**: Análisis exploratorio detallado de la distribución de clases, solapamiento espacial y estructura morfológica inicial.

---
⚠️ **Nota sobre Reproducibilidad**: Los datos crudos dentro de las carpetas de origen no deben modificarse manualmente para asegurar la integridad de los experimentos comparados en la tesis.

---
[🏠 Volver al Inicio](../README.md) | [Siguiente: Scripts ➔](../scripts/README.md)

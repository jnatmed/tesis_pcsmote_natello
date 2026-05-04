[⬅️ Previo: Scripts](../scripts/README.md) | [Siguiente: Resultados ➜](../resultados/README.md) | [🏠 Inicio](../README.md)

# Notebooks de Experimentación y Evaluación

Esta carpeta contiene los notebooks de Jupyter utilizados para documentar la experimentación de PC-SMOTE frente a técnicas de sobremuestreo clásicas y extensiones complementarias.

El pipeline final de la tesis se basa en una comparación con **Random Forest bajo hiperparámetros fijos**. Esta decisión metodológica busca aislar el efecto del sobremuestreo: lo que cambia entre condiciones es el conjunto de entrenamiento generado por cada técnica, no la configuración del clasificador.

## Análisis Destacados

- **`01_grid_pc_smote.ipynb`**: exploración inicial de configuraciones de PC-SMOTE y calibración heurística basada en percentiles.
- **`02_experimento_params_fijos.ipynb`**: notebook principal de evaluación final con `Random Forest` usando hiperparámetros fijos. Compara el caso base, técnicas clásicas (`SMOTE`, `Borderline-SMOTE`, `ADASYN`) y `PC-SMOTE` sobre datasets como Shuttle, Telco Churn, US Crime y Predict Faults.
- **`03_experimento_modelos_alternativos_params_fijos.ipynb`**: extensión experimental con modelos alternativos y datasets generados por técnicas contemporáneas. Mantiene la lógica de parámetros fijos para preservar comparabilidad.

## Criterio experimental

La comparación final no optimiza hiperparámetros del clasificador por separado para cada versión sobremuestreada. En su lugar:

1. Se separa `train/test`.
2. Se aplica escalado y, según el caso, limpieza opcional sobre `train`.
3. Se generan versiones de entrenamiento mediante técnicas clásicas y PC-SMOTE.
4. Todas las versiones se evalúan con el mismo clasificador base y sobre el mismo `test` preservado.

De este modo, la evaluación busca medir el impacto de la estrategia de sobremuestreo y evitar que las diferencias se mezclen con ajustes particulares del modelo.

---

### Instrucciones de Ejecución

Asegúrese de tener instalados los siguientes entornos y dependencias antes de ejecutar los notebooks:

1. Python 3.12+.
2. `scikit-learn`, `pandas`, `numpy`, `matplotlib` y `seaborn`.
3. Acceso a los scripts auxiliares de la carpeta `scripts/` mediante `PYTHONPATH` o ejecutando desde la raíz del proyecto.

---

[⬅️ Previo: Scripts](../scripts/README.md) | [Siguiente: Resultados ➜](../resultados/README.md) | [🏠 Inicio](../README.md)

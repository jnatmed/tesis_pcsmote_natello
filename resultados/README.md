[⬅️ Previo: Notebooks](../notebooks/README.md) | [🏠 Volver al Inicio](../README.md)

# Resultados y Métricas de Evaluación

En este directorio se almacenan los productos de la fase experimental de PC-SMOTE, incluyendo registros detallados, tablas comparativas y mejores configuraciones encontradas.

## Principales Reportes

- **`log_resultados_*.txt`**: registros detallados de la ejecución de cada experimento de sobremuestreo y clasificación sobre los distintos conjuntos de datos.
- **`resultados_RS_cv_vs_test*.xlsx`**: hojas de cálculo que comparan el desempeño del clasificador (`Random Forest`) en validación cruzada frente al conjunto de prueba, detallando métricas como F1-macro y recall.
- **`best_params_por_contexto.json`**: configuraciones de hiperparámetros óptimas descubiertas mediante la búsqueda sistemática para cada escenario.
- **`analisis_tabla_entropia.xlsx`**: tabulación de los efectos del filtrado por entropía en la selección de semillas para PC-SMOTE.

---

### Interpretación de Métricas

Los resultados aquí documentados se centran en el **F1-macro average** como métrica rectora y permiten comparar el comportamiento de PC-SMOTE frente a técnicas clásicas, tanto en escenarios donde mejora el rendimiento como en aquellos donde restringe la generación de sintéticos porque la vecindad no resulta favorable.

---

[⬅️ Previo: Notebooks](../notebooks/README.md) | [🏠 Volver al Inicio](../README.md)

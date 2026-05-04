[⬅️ Previo: Notebooks](../notebooks/README.md) | [🏠 Volver al Inicio](../README.md)

# Resultados y Métricas de Evaluación

En este directorio se almacenan los productos de la fase experimental de PC-SMOTE, incluyendo registros detallados, tablas comparativas, configuraciones seleccionadas y resultados complementarios.

Los resultados finales de la tesis se interpretan a partir de un pipeline con **Random Forest de hiperparámetros fijos**. Esto permite comparar el efecto de cada versión del conjunto de entrenamiento: caso base, técnicas clásicas de sobremuestreo y PC-SMOTE.

## Principales Reportes

- **`log_resultados_*.txt`**: registros detallados de la ejecución de cada experimento de sobremuestreo y clasificación sobre los distintos conjuntos de datos.
- **`resultados_params_fijos.xlsx`**: resultados principales con `Random Forest` bajo hiperparámetros fijos.
- **`resultados_modelos_alternativos_params_fijos.xlsx`**: extensión con modelos alternativos manteniendo la lógica de parámetros fijos.
- **`resultados_params_fijos_vs_contemporaneos.xlsx`**: comparación complementaria que incorpora técnicas contemporáneas de sobremuestreo.
- **`resultados_RS_cv_vs_test*.xlsx`**: hojas de cálculo de etapas previas o auxiliares que comparan desempeño en validación cruzada frente al conjunto de prueba.
- **`best_params_por_contexto.json`**: configuraciones seleccionadas durante la exploración experimental de PC-SMOTE.
- **`analisis_tabla_entropia.xlsx`**: tabulación de los efectos del filtrado por entropía en la selección de semillas para PC-SMOTE.
- **`validacion_pipeline_contemporaneos_y_modelos.md`**: lectura complementaria de la extensión con modelos alternativos y técnicas contemporáneas.

---

### Interpretación de Métricas

Los resultados aquí documentados se centran en el **F1-macro average** como métrica rectora y permiten comparar el comportamiento de PC-SMOTE frente a técnicas clásicas, tanto en escenarios donde mejora el rendimiento como en aquellos donde restringe la generación de sintéticos porque la vecindad no resulta favorable.

Las métricas se leen sobre el conjunto de test preservado. El sobremuestreo y la limpieza opcional con `Isolation Forest` se aplican sobre entrenamiento, para evitar fuga de información.

---

[⬅️ Previo: Notebooks](../notebooks/README.md) | [🏠 Volver al Inicio](../README.md)

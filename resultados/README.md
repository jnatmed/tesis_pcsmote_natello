[⬅️ Previo: Notebooks](../notebooks/README.md) | [🏠 Volver al Inicio](../README.md)

# 📈 Resultados y Métricas de Evaluación

En este directorio se almacenan los productos de la fase experimental de PC-SMOTE, incluyendo registros detallados, tablas comparativas y mejores configuraciones de parámetros encontradas.

## 📊 Principales Reportes

- **`log_resultados_*.txt`**: Registros detallados de la ejecución de cada experimento de sobremuestreo y clasificación aplicada sobre los diversos conjuntos de datos.
- **`resultados_RS_cv_vs_test*.xlsx`**: Hojas de cálculo que comparan el desempeño del clasificador (*Random Forest*) en la validación cruzada frente al conjunto de prueba, detallando métricas clave como F1-macro y Recall.
- **`best_params_por_contexto.json`**: Almacena las configuraciones de hiperparámetros óptimas descubiertas mediante la búsqueda sistemática para cada escenario (dataset y técnica).
- **`analisis_tabla_entropia.xlsx`**: Tabulación de los efectos del filtrado por entropía en la selección de semillas para PC-SMOTE (Capítulo 4 de la tesis).

---
### 🛠️ Interpretación de Métricas
Los resultados aquí documentados se centran en el **F1-macro average** como métrica rectora, diseñada para valorar equitativamente el desempeño predictivo en ambas clases y evidenciar la eficacia de PC-SMOTE frente a SMOTE convencional en entornos de gran solapamiento estructural.

---
[⬅️ Previo: Notebooks](../notebooks/README.md) | [🏠 Volver al Inicio](../README.md)

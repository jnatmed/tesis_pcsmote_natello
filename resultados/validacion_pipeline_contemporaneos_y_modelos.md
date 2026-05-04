# Validación del pipeline experimental extendido

## Objetivo

Este documento resume cómo validar y explicar el pipeline experimental que incorpora:

- Técnicas contemporáneas de sobremuestreo: `RadiusSMOTE`, `LD-SMOTE` y `VS-SMOTE`.
- Modelos alternativos de clasificación: `XGBoost`, `SVM`, `Naive Bayes` y `Logistic Regression`.
- Métricas adicionales: `AvAcc` y tiempos de ejecución.

La idea central para defender ante el tutor es: todos los clasificadores se ejecutan con hiperparámetros fijos y todos los métodos de sobremuestreo se aplican solo sobre entrenamiento. Por lo tanto, las diferencias observadas se atribuyen al conjunto de entrenamiento generado por cada técnica, no al ajuste del clasificador.

## Archivos principales

- Notebook Random Forest original: `notebooks/02_experimento_params_fijos.ipynb`
- Notebook modelos alternativos: `notebooks/03_experimento_modelos_alternativos_params_fijos.ipynb`
- Resultados modelos alternativos: `resultados/resultados_modelos_alternativos_params_fijos.xlsx`
- Generador de datasets contemporáneos: `scripts/generar_contemporaneos.py`
- Base común multiclase: `scripts/contemporaneos/sampler_contemporaneo.py`
- Implementaciones: `scripts/contemporaneos/radius_smote.py`, `scripts/contemporaneos/ld_smote.py`, `scripts/contemporaneos/vs_smote.py`

## Diseño validado

El flujo mantiene la misma lógica experimental de la tesis:

1. Se parte de CSV base ya particionados en `train` y `test`.
2. Las técnicas de sobremuestreo se aplican exclusivamente sobre `train`.
3. Los archivos `test` no se sobremuestrean ni se limpian.
4. Cada técnica genera datasets balanceados hasta igualar la clase mayoritaria.
5. Cada modelo se entrena sobre una versión del `train`.
6. Cada modelo se evalúa sobre el mismo `test` preservado.
7. El delta se calcula contra el baseline del mismo dataset, mismo grado de limpieza y mismo modelo.

Esto evita fuga de información y mantiene una comparación justa.

## Validación mecánica realizada

Se validaron 30 archivos contemporáneos generados:

- 10 con `LD-SMOTE`.
- 10 con `Radius-SMOTE`.
- 10 con `VS-SMOTE`.

Controles realizados:

- El número de sintéticos indicado en el nombre del archivo coincide con el número real de filas agregadas.
- Todas las salidas quedan balanceadas por clase.
- Los datasets contemporáneos se generan desde `datasets/datasets_aumentados/base/`.
- El pipeline de notebooks toma esos datasets desde `datasets/datasets_aumentados/contemporaneos/`.

Resultado del control:

- `sg_ok = True` para los 30 archivos.
- `balanced_out = True` para los 30 archivos.

## Implementación de técnicas contemporáneas

### Base común

`SamplerContemporaneo` implementa una base `one-vs-rest` para soportar datasets binarios y multiclase. Cada técnica implementa su lógica binaria y la clase base la aplica clase contra resto.

Esto permite usar las mismas técnicas en datasets binarios como `US Crime` y en multiclase como `Shuttle`.

### Radius-SMOTE

Implementación práctica:

- Selecciona semillas minoritarias seguras mediante KNN.
- Calcula la mayoría más cercana para cada semilla.
- Genera sintéticos alrededor de la semilla usando la dirección hacia la frontera.
- Si no hay semillas seguras, usa las minoritarias disponibles como respaldo.

Punto para defender: es una implementación operativa basada en el paper, adaptada al pipeline multiclase y al formato `fit_resample`.

### LD-SMOTE

Implementación práctica:

- Calcula grados de contribución por atributo mediante partición simple con `KMeans`.
- Usa distancias ponderadas por esos grados.
- Estima densidad local.
- Distribuye la cantidad de sintéticos según densidad.
- Genera dentro de triángulos formados por semilla y vecinos minoritarios.

Punto para defender: se preserva la idea principal del paper: densidad local y generación geométrica dentro de regiones minoritarias.

### VS-SMOTE

Implementación práctica:

- Construye líneas entre semillas minoritarias y vecinos minoritarios.
- Evalúa el punto medio de cada línea con KNN global.
- Usa líneas cuyo vecindario medio tiene suficiente presencia minoritaria.
- Si no aparecen líneas válidas, puede usar todas las líneas minoritarias como respaldo operativo.

Punto para defender: el centro del método es seleccionar líneas de alto valor antes de interpolar; el respaldo evita fallas de ejecución en datasets donde el filtro queda vacío.

## Modelos alternativos

### XGBoost

Referencia: Malyarchuk usa XGBoost sobre Shuttle y reporta el modelo como baseline fuerte. El paper menciona `xgboost`, `max_depth=6` y `learning_rate=0.3` como valores default no exhaustivamente optimizados.

Configuración fija usada:

- `n_estimators=100`
- `max_depth=6`
- `learning_rate=0.3`
- `tree_method="hist"`
- `random_state=42`
- `n_jobs=1`

El objetivo se define dinámicamente:

- Binario: `binary:logistic`
- Multiclase: `multi:softprob`

### Logistic Regression, SVM y Naive Bayes

Referencia: Sonoda et al. usan Logistic Regression, SVM y Naive Bayes desde scikit-learn con parámetros default y parámetros fijos para todas las corridas.

Configuración fija usada:

- `SVM`: `SVC(C=1.0, kernel="rbf", gamma="scale")`
- `NaiveBayes`: `GaussianNB()`
- `LogisticRegression`: `penalty="l2"`, `C=1.0`, `solver="lbfgs"`, `max_iter=1000`

Nota: `max_iter=1000` se fija para asegurar convergencia, sin hacer tuning.

## Métrica AvAcc

Se agregó `AvAcc` como `balanced_accuracy_score`.

Interpretación:

- En binario equivale a `(TPR + TNR) / 2`.
- En multiclase equivale al promedio del recall por clase.

Esto es útil en desbalance porque no deja que la clase mayoritaria oculte el mal rendimiento en minoritarias.

## Resultados principales de modelos alternativos

Mejor resultado por dataset y modelo:

| Dataset | Modelo | Mejor técnica | Limpieza | F1-macro | Delta vs baseline mismo modelo | AvAcc | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| Shuttle | XGBoost | VS-SMOTE | 0 | 1.0000 | +0.0499 | 1.0000 | 1.0000 |
| US Crime | XGBoost | PC-SMOTE | 0 | 0.7895 | +0.0242 | 0.7851 | 0.9424 |
| US Crime | SVM | PC-SMOTE | 1 | 0.7452 | +0.2060 | 0.7603 | 0.9248 |
| US Crime | Naive Bayes | PC-SMOTE | 3 | 0.7710 | +0.0973 | 0.8215 | 0.9248 |
| US Crime | Logistic Regression | BorderlineSMOTE | 0 | 0.7742 | +0.0416 | 0.8913 | 0.9123 |

Lectura importante:

- PC-SMOTE gana con XGBoost, SVM y Naive Bayes en `US Crime`.
- Logistic Regression favorece BorderlineSMOTE.
- En `Shuttle`, XGBoost favorece VS-SMOTE con rendimiento perfecto en esta partición.

## Tiempos de ejecución

La medición de tiempos quedó incorporada en el Excel con cuatro columnas:

- `tiempo_carga_seg`
- `tiempo_entrenamiento_seg`
- `tiempo_prediccion_seg`
- `tiempo_total_seg`

Resumen por dataset y modelo:

| Dataset | Modelo | Corridas | Tiempo medio total | Mediana | Mínimo | Máximo |
|---|---:|---:|---:|---:|---:|---:|
| Shuttle | XGBoost | 18 | 9.578 s | 10.518 s | 1.678 s | 13.447 s |
| US Crime | XGBoost | 38 | 0.478 s | 0.500 s | 0.196 s | 0.557 s |
| US Crime | SVM | 38 | 0.147 s | 0.154 s | 0.042 s | 0.263 s |
| US Crime | Logistic Regression | 38 | 0.325 s | 0.335 s | 0.187 s | 0.415 s |
| US Crime | Naive Bayes | 38 | 0.003 s | 0.003 s | 0.002 s | 0.004 s |

Lectura:

- El costo dominante aparece en `Shuttle + XGBoost`, porque los conjuntos aumentados llegan a más de 250 mil filas.
- En `US Crime`, todos los modelos son baratos computacionalmente; incluso XGBoost queda por debajo de 0.6 segundos por corrida.
- `Naive Bayes` es prácticamente instantáneo, lo cual es coherente con su simplicidad.
- En Shuttle, los métodos que más tardan son los que entrenan XGBoost sobre datasets aumentados grandes. La corrida más lenta fue `Radius-SMOTE` sin limpieza, con 255.283 filas y 13.447 segundos.

Tiempos medios por familia en `Shuttle + XGBoost`:

| Tipo | Corridas | Tiempo medio total |
|---|---:|---:|
| Base | 2 | 1.701 s |
| Clásico | 6 | 9.987 s |
| Contemporáneo | 6 | 11.616 s |
| PC-SMOTE | 4 | 9.845 s |

Tiempos medios por familia en `US Crime + XGBoost`:

| Tipo | Corridas | Tiempo medio total |
|---|---:|---:|
| Base | 3 | 0.202 s |
| Clásico | 9 | 0.531 s |
| Contemporáneo | 9 | 0.524 s |
| PC-SMOTE | 17 | 0.473 s |

Punto para defender:

Los tiempos medidos corresponden al entrenamiento y predicción de los modelos sobre datasets ya generados. No miden el costo de generación del sobremuestreo contemporáneo o PC-SMOTE. Por eso sirven para comparar el costo de entrenar modelos sobre distintos tamaños de `train`, pero no para comparar directamente el costo interno de cada algoritmo de sobremuestreo.

## Qué se puede defender

El experimento extendido aporta robustez porque PC-SMOTE no queda validado solo con Random Forest. En `US Crime`, que es uno de los datasets más difíciles por alta dimensionalidad y escasez minoritaria, PC-SMOTE mejora con tres clasificadores distintos.

También aporta comparación externa porque se agregan técnicas contemporáneas implementadas desde papers. Eso permite discutir PC-SMOTE no solo contra SMOTE, Borderline-SMOTE y ADASYN, sino contra propuestas más recientes.

## Qué conviene reconocer como limitación

- Las implementaciones contemporáneas son recreaciones operativas basadas en los papers, no librerías oficiales de los autores.
- Algunas decisiones prácticas se adaptaron al pipeline: soporte multiclase `one-vs-rest`, respaldo cuando no hay semillas/líneas válidas y normalización del formato `fit_resample`.
- No hay ajuste de hiperparámetros. Esto es deliberado para aislar el efecto del sobremuestreo, pero puede dejar a algunos modelos por debajo de su rendimiento máximo posible.
- Sonoda usa validación cruzada; este pipeline usa partición train/test fija para mantener coherencia con la tesis. Una extensión futura sería repetir todo con validación cruzada estratificada.

## Cómo reproducir

Generar contemporáneos:

```powershell
python scripts/generar_contemporaneos.py --overwrite --random-state 42
```

Ejecutar Random Forest:

```powershell
jupyter nbconvert --to notebook --execute notebooks/02_experimento_params_fijos.ipynb --output 02_experimento_params_fijos.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=7200
```

Ejecutar modelos alternativos:

```powershell
jupyter nbconvert --to notebook --execute notebooks/03_experimento_modelos_alternativos_params_fijos.ipynb --output 03_experimento_modelos_alternativos_params_fijos.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=7200
```

## Guion corto para tutor

Se extendió el pipeline original de Random Forest manteniendo la misma regla metodológica: hiperparámetros fijos y evaluación sobre test preservado. Además de los métodos clásicos, se incorporaron tres técnicas contemporáneas implementadas a partir de papers: Radius-SMOTE, LD-SMOTE y VS-SMOTE. Para comparar con literatura, se agregó XGBoost tomando como referencia Malyarchuk en Shuttle, y Logistic Regression, SVM y Naive Bayes tomando como referencia Sonoda en US Crime. El resultado es que PC-SMOTE no solo funciona con Random Forest: en US Crime también obtiene el mejor F1-macro con XGBoost, SVM y Naive Bayes. Eso refuerza que la heurística geométrica propuesta tiene valor más allá de un único clasificador.

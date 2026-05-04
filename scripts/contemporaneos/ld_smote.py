from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from .sampler_contemporaneo import SamplerContemporaneo


class LDSMOTE(SamplerContemporaneo):
    """
    LD-SMOTE binario con:
      - estimacion de densidad local basada en contribution degrees
      - distribucion de sinteticos segun densidad normalizada
      - generacion dentro de triangulos formados por una semilla y dos vecinos
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        sampling_strategy: str = "auto",
        random_state=None,
        epsilon: float = 1e-12,
    ):
        super().__init__(sampling_strategy=sampling_strategy, random_state=random_state)
        self.k_neighbors = int(k_neighbors)
        self.epsilon = float(epsilon)

    def _fit_resample_binario(
        self,
        X_min: np.ndarray,
        X_maj: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        if n_samples <= 0 or len(X_min) == 0:
            return np.empty((0, X_min.shape[1]), dtype=float)
        if len(X_min) == 1:
            return np.repeat(X_min, repeats=n_samples, axis=0)

        X_full = np.vstack([X_min, X_maj])
        y_full = np.concatenate(
            [
                np.ones(len(X_min), dtype=int),
                np.zeros(len(X_maj), dtype=int),
            ]
        )

        contribution_degrees = self._calcular_contribution_degrees(X_full, y_full)
        matriz_distancias = self._matriz_distancias_ponderadas(X_min, contribution_degrees)

        k_base = min(self.k_neighbors, len(X_min) - 1)
        if k_base <= 0:
            return np.repeat(X_min, repeats=n_samples, axis=0)

        vecinos_ordenados = np.argsort(matriz_distancias, axis=1)
        vecinos_base = vecinos_ordenados[:, :k_base]

        distancias_base = np.take_along_axis(matriz_distancias, vecinos_base, axis=1)
        distancias_promedio = distancias_base.mean(axis=1)

        mediana_ad = float(np.median(distancias_promedio))
        if mediana_ad <= self.epsilon:
            densidades = np.ones(len(X_min), dtype=float)
        else:
            densidades = np.exp(
                -((distancias_promedio ** 2) / (2.0 * (mediana_ad ** 2)))
            )

        pesos = densidades / max(densidades.sum(), self.epsilon)
        cuotas = self._asignar_cantidades_por_pesos(pesos, n_samples)
        promedio_ad = float(np.mean(distancias_promedio))

        sinteticos = []
        for indice_semilla, cantidad in enumerate(cuotas):
            if cantidad <= 0:
                continue

            k_local = k_base
            if (
                distancias_promedio[indice_semilla] > promedio_ad
                and len(X_min) - 1 >= (k_base + 1)
            ):
                k_local = k_base + 1

            pool = vecinos_ordenados[indice_semilla, :k_local]
            if len(pool) == 0:
                continue

            semilla = X_min[indice_semilla]
            for _ in range(int(cantidad)):
                if len(pool) >= 2:
                    vecinos = self._rng.choice(pool, size=2, replace=False)
                    vecino_a = X_min[int(vecinos[0])]
                    vecino_b = X_min[int(vecinos[1])]
                    alpha, beta = self._samplear_punto_en_triangulo(self._rng)
                    nuevo = semilla + alpha * (vecino_a - semilla) + beta * (vecino_b - semilla)
                else:
                    vecino = X_min[int(pool[0])]
                    alpha = float(self._rng.uniform(0.0, 1.0))
                    nuevo = semilla + alpha * (vecino - semilla)

                sinteticos.append(nuevo)

        if not sinteticos:
            return np.empty((0, X_min.shape[1]), dtype=float)

        return np.asarray(sinteticos, dtype=float)

    def _calcular_contribution_degrees(
        self,
        X_full: np.ndarray,
        y_full: np.ndarray,
    ) -> np.ndarray:
        positivos = y_full == 1
        cantidad_features = X_full.shape[1]
        contribution_degrees = np.ones(cantidad_features, dtype=float)

        for j in range(cantidad_features):
            columna = X_full[:, j].reshape(-1, 1)

            if np.unique(columna).size < 2:
                contribution_degrees[j] = 1.0
                continue

            modelo = KMeans(n_clusters=2, n_init=10, random_state=self.random_state)
            clusters = modelo.fit_predict(columna)

            mejor_jaccard = self.epsilon
            for cluster_id in (0, 1):
                mascara_cluster = clusters == cluster_id
                union = np.logical_or(positivos, mascara_cluster).sum()
                if union == 0:
                    continue

                interseccion = np.logical_and(positivos, mascara_cluster).sum()
                jaccard = interseccion / union
                mejor_jaccard = max(mejor_jaccard, float(jaccard))

            contribution_degrees[j] = max(mejor_jaccard, self.epsilon)

        return contribution_degrees

    def _matriz_distancias_ponderadas(
        self,
        X_min: np.ndarray,
        contribution_degrees: np.ndarray,
    ) -> np.ndarray:
        diffs = np.abs(X_min[:, None, :] - X_min[None, :, :])
        ponderado = diffs / contribution_degrees.reshape(1, 1, -1)
        matriz = ponderado.mean(axis=2)
        np.fill_diagonal(matriz, np.inf)
        return matriz

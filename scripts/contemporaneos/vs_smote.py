from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .sampler_contemporaneo import SamplerContemporaneo


class VSSMOTE(SamplerContemporaneo):
    """
    VS-SMOTE binario.

    Usa lineas de conexion entre una semilla minoritaria y sus vecinos
    minoritarios; cada linea se evalua con un middle sample y solo se usan
    las de alto valor para generar sinteticos.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        k_safe: int = 3,
        k_high: int = 5,
        sampling_strategy: str = "auto",
        random_state=None,
        fallback_to_all_lines: bool = True,
    ):
        super().__init__(sampling_strategy=sampling_strategy, random_state=random_state)
        self.k_neighbors = int(k_neighbors)
        self.k_safe = int(k_safe)
        self.k_high = int(k_high)
        self.fallback_to_all_lines = bool(fallback_to_all_lines)

        if not 0 <= self.k_safe <= self.k_high:
            raise ValueError("Se requiere 0 <= k_safe <= k_high.")

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

        X_all = np.vstack([X_min, X_maj])
        y_all = np.concatenate(
            [
                np.ones(len(X_min), dtype=int),
                np.zeros(len(X_maj), dtype=int),
            ]
        )

        lineas = self._seleccionar_lineas_de_alto_valor(X_min, X_all, y_all)
        if not lineas and self.fallback_to_all_lines:
            lineas = self._todas_las_lineas(X_min)

        if not lineas:
            return np.empty((0, X_min.shape[1]), dtype=float)

        sinteticos = np.empty((n_samples, X_min.shape[1]), dtype=float)
        for i in range(n_samples):
            origen, destino = lineas[int(self._rng.randint(0, len(lineas)))]
            alpha = float(self._rng.uniform(0.0, 1.0))
            sinteticos[i] = origen + alpha * (destino - origen)

        return sinteticos

    def _seleccionar_lineas_de_alto_valor(
        self,
        X_min: np.ndarray,
        X_all: np.ndarray,
        y_all: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        k_vecinos = min(self.k_neighbors, len(X_min) - 1)
        if k_vecinos <= 0:
            return []

        knn_min = NearestNeighbors(n_neighbors=k_vecinos + 1)
        knn_min.fit(X_min)
        indices_min = knn_min.kneighbors(X_min, return_distance=False)[:, 1:]

        knn_all = NearestNeighbors(n_neighbors=min(self.k_neighbors, len(X_all)))
        knn_all.fit(X_all)

        lineas = []
        for idx_semilla, vecinos in enumerate(indices_min):
            semilla = X_min[idx_semilla]
            for idx_vecino in vecinos:
                vecino = X_min[int(idx_vecino)]
                medio = 0.5 * (semilla + vecino)
                indices_all = knn_all.kneighbors(medio.reshape(1, -1), return_distance=False)[0]
                cantidad_minoritarios = int((y_all[indices_all] == 1).sum())

                if self.k_safe <= cantidad_minoritarios <= self.k_high:
                    lineas.append((semilla, vecino))

        return lineas

    def _todas_las_lineas(
        self,
        X_min: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        k_vecinos = min(self.k_neighbors, len(X_min) - 1)
        if k_vecinos <= 0:
            return []

        knn = NearestNeighbors(n_neighbors=k_vecinos + 1)
        knn.fit(X_min)
        indices = knn.kneighbors(X_min, return_distance=False)[:, 1:]

        lineas = []
        for idx_semilla, vecinos in enumerate(indices):
            semilla = X_min[idx_semilla]
            for idx_vecino in vecinos:
                lineas.append((semilla, X_min[int(idx_vecino)]))
        return lineas

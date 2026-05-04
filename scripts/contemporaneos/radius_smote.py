from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from .sampler_contemporaneo import SamplerContemporaneo


class RadiusSMOTE(SamplerContemporaneo):
    """
    Implementacion practica de Radius-SMOTE para el caso binario.

    Supuestos operativos tomados del paper:
      - Se filtran semillas minoritarias "seguras" con KNN.
      - El radio seguro se define por la distancia a la mayoria mas cercana.
      - La sintesis ocurre sobre la recta que une la semilla con esa mayoria,
        usando ambos sentidos de interpolacion.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        sampling_strategy: str = "auto",
        random_state=None,
    ):
        super().__init__(sampling_strategy=sampling_strategy, random_state=random_state)
        self.k_neighbors = int(k_neighbors)

    def _fit_resample_binario(
        self,
        X_min: np.ndarray,
        X_maj: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        if n_samples <= 0 or len(X_min) == 0 or len(X_maj) == 0:
            return np.empty((0, X_min.shape[1]), dtype=float)

        semillas = self._seleccionar_semillas_seguras(X_min, X_maj)
        if len(semillas) == 0:
            semillas = X_min

        buscador_mayoria = NearestNeighbors(n_neighbors=1)
        buscador_mayoria.fit(X_maj)
        _, indices_may = buscador_mayoria.kneighbors(semillas)
        mayoria_cercana = X_maj[indices_may[:, 0]]

        sinteticos = np.empty((n_samples, X_min.shape[1]), dtype=float)

        for i in range(n_samples):
            idx = int(self._rng.randint(0, len(semillas)))
            semilla = semillas[idx]
            frontera = mayoria_cercana[idx]
            direccion = frontera - semilla

            if np.allclose(direccion, 0.0):
                sinteticos[i] = semilla.copy()
                continue

            signo = -1.0 if self._rng.uniform(0.0, 1.0) < 0.5 else 1.0
            factor = float(self._rng.uniform(0.0, 1.0 - 1e-6))
            sinteticos[i] = semilla + (signo * factor * direccion)

        return sinteticos

    def _seleccionar_semillas_seguras(
        self,
        X_min: np.ndarray,
        X_maj: np.ndarray,
    ) -> np.ndarray:
        X_all = np.vstack([X_min, X_maj])
        y_all = np.concatenate(
            [
                np.ones(len(X_min), dtype=int),
                np.zeros(len(X_maj), dtype=int),
            ]
        )

        vecinos = min(self.k_neighbors, len(X_all))
        if vecinos < 1:
            return X_min

        clasificador = KNeighborsClassifier(n_neighbors=vecinos)
        clasificador.fit(X_all, y_all)
        pred = clasificador.predict(X_min)
        mascara_segura = pred == 1
        return X_min[mascara_segura]

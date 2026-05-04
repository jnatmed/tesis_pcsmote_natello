from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from sklearn.utils import check_random_state


@dataclass(frozen=True)
class ObjetivoClase:
    clase: object
    cantidad_actual: int
    cantidad_objetivo: int

    @property
    def cantidad_a_generar(self) -> int:
        return max(0, self.cantidad_objetivo - self.cantidad_actual)


class SamplerContemporaneo(ABC):
    """
    Base comun para samplers contemporaneos.

    Cada subclase implementa SOLO el caso binario:
      - X_min: muestras de la clase objetivo
      - X_maj: muestras del resto de las clases
      - n_samples: cantidad exacta de sinteticos a generar

    El soporte multiclase se resuelve aqui via one-vs-rest, de forma que
    el pipeline pueda trabajar tambien con datasets como shuttle.
    """

    def __init__(self, sampling_strategy: str = "auto", random_state=None):
        if sampling_strategy != "auto":
            raise ValueError(
                "Por ahora sampling_strategy solo admite el valor 'auto'."
            )

        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self._rng = check_random_state(random_state)

    def fit_resample(self, X, y):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)

        if X_arr.ndim != 2:
            raise ValueError("X debe ser una matriz 2D.")
        if len(X_arr) != len(y_arr):
            raise ValueError("X e y deben tener la misma cantidad de filas.")
        if len(X_arr) == 0:
            raise ValueError("No se puede sobremuestrear un dataset vacio.")

        objetivos = self._resolver_objetivos_multiclase(y_arr)
        if not objetivos:
            return X_arr.copy(), y_arr.copy()

        sinteticos_X = []
        sinteticos_y = []

        for objetivo in objetivos:
            cantidad = objetivo.cantidad_a_generar
            if cantidad <= 0:
                continue

            mascara_min = y_arr == objetivo.clase
            X_min = X_arr[mascara_min]
            X_maj = X_arr[~mascara_min]

            X_syn = self._fit_resample_binario(
                X_min=X_min,
                X_maj=X_maj,
                n_samples=cantidad,
            )

            if X_syn.size == 0:
                continue

            sinteticos_X.append(X_syn)
            sinteticos_y.append(np.full(len(X_syn), objetivo.clase, dtype=y_arr.dtype))

        if not sinteticos_X:
            return X_arr.copy(), y_arr.copy()

        X_out = np.vstack([X_arr, *sinteticos_X])
        y_out = np.concatenate([y_arr, *sinteticos_y])
        return X_out, y_out

    def _resolver_objetivos_multiclase(self, y: np.ndarray) -> list[ObjetivoClase]:
        clases, conteos = np.unique(y, return_counts=True)
        if len(clases) < 2:
            return []

        mayoritaria = int(np.max(conteos))
        objetivos = []

        for clase, conteo in zip(clases, conteos):
            if int(conteo) >= mayoritaria:
                continue
            objetivos.append(
                ObjetivoClase(
                    clase=clase,
                    cantidad_actual=int(conteo),
                    cantidad_objetivo=mayoritaria,
                )
            )

        return objetivos

    @staticmethod
    def _asignar_cantidades_por_pesos(pesos: np.ndarray, total: int) -> np.ndarray:
        pesos = np.asarray(pesos, dtype=float)
        if total <= 0 or pesos.size == 0:
            return np.zeros(len(pesos), dtype=int)

        suma = pesos.sum()
        if suma <= 0:
            base = np.full(len(pesos), total // len(pesos), dtype=int)
            base[: total % len(pesos)] += 1
            return base

        cuotas = total * (pesos / suma)
        enteros = np.floor(cuotas).astype(int)
        faltan = total - int(enteros.sum())

        if faltan > 0:
            residuos = cuotas - enteros
            orden = np.argsort(-residuos)
            enteros[orden[:faltan]] += 1

        return enteros

    @staticmethod
    def _samplear_punto_en_triangulo(rng) -> tuple[float, float]:
        alpha = float(rng.uniform(0.0, 1.0))
        beta = float(rng.uniform(0.0, 1.0))
        if alpha + beta >= 1.0:
            alpha = 1.0 - alpha
            beta = 1.0 - beta
        return alpha, beta

    @abstractmethod
    def _fit_resample_binario(
        self,
        X_min: np.ndarray,
        X_maj: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        raise NotImplementedError

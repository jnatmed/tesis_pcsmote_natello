from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from contemporaneos import LDSMOTE, RadiusSMOTE, VSSMOTE


PATRON_BASE = re.compile(r"(?P<dataset>.+?)_I(?P<grado>\d+)_tm(?P<tm>\d+)_train\.csv$")
SCRIPT_DIR = Path(__file__).resolve().parent


def inferir_columna_target(df: pd.DataFrame) -> str:
    return "target" if "target" in df.columns else df.columns[-1]


def construir_sampler(nombre: str, random_state: int):
    nombre = nombre.lower()
    if nombre == "radius-smote":
        return RadiusSMOTE(random_state=random_state)
    if nombre == "ld-smote":
        return LDSMOTE(random_state=random_state)
    if nombre == "vs-smote":
        return VSSMOTE(random_state=random_state)
    raise ValueError(f"Tecnica no soportada: {nombre}")


def resolver_ruta(valor: str) -> Path:
    ruta = Path(valor)
    if ruta.is_absolute():
        return ruta
    return (SCRIPT_DIR / ruta).resolve()


def generar_desde_base(
    ruta_base: Path,
    ruta_salida: Path,
    tecnicas: list[str],
    overwrite: bool,
    random_state: int,
    datasets_permitidos: set[str] | None,
):
    ruta_salida.mkdir(parents=True, exist_ok=True)
    archivos = sorted(ruta_base.glob("*_train.csv"))

    if not archivos:
        raise FileNotFoundError(f"No se encontraron CSV base en: {ruta_base}")

    for archivo in archivos:
        match = PATRON_BASE.match(archivo.name)
        if match is None:
            continue

        dataset = match.group("dataset")
        grado = match.group("grado")

        if datasets_permitidos is not None and dataset not in datasets_permitidos:
            continue

        df = pd.read_csv(archivo)
        target = inferir_columna_target(df)
        columnas = [c for c in df.columns if c != target]
        X = df[columnas].to_numpy(dtype=float, copy=False)
        y = df[target].to_numpy()

        print(f"\nDataset base: {archivo.name}")
        print(f"  target: {target}")
        print(f"  clases: {sorted(pd.Series(y).unique().tolist())}")
        print(f"  filas originales: {len(df)}")

        for tecnica in tecnicas:
            existentes = list(ruta_salida.glob(f"{tecnica}_{dataset}_I{grado}_sg*_train.csv"))
            if existentes and not overwrite:
                print(f"  - {tecnica}: omitido, ya existe {existentes[0].name}")
                continue

            sampler = construir_sampler(tecnica, random_state=random_state)
            X_res, y_res = sampler.fit_resample(X, y)

            sinteticos = len(X_res) - len(X)
            salida_final = ruta_salida / f"{tecnica}_{dataset}_I{grado}_sg{sinteticos}_train.csv"

            df_res = pd.DataFrame(X_res, columns=columnas)
            df_res[target] = y_res
            df_res.to_csv(salida_final, index=False)

            print(
                f"  - {tecnica}: guardado {salida_final.name} "
                f"(sg={sinteticos}, total={len(df_res)})"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera datasets de sobremuestreo contemporaneo a partir de los CSV base."
    )
    parser.add_argument(
        "--base-dir",
        default="../datasets/datasets_aumentados/base",
        help="Carpeta con los CSV base de entrenamiento.",
    )
    parser.add_argument(
        "--output-dir",
        default="../datasets/datasets_aumentados/contemporaneos",
        help="Carpeta de salida para los CSV contemporaneos.",
    )
    parser.add_argument(
        "--techniques",
        nargs="+",
        default=["radius-smote", "ld-smote", "vs-smote"],
        help="Tecnicas a ejecutar.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Subset de datasets logicos a procesar.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescribe archivos existentes.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla aleatoria para todos los samplers.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    datasets_permitidos = None if args.datasets is None else set(args.datasets)

    generar_desde_base(
        ruta_base=resolver_ruta(args.base_dir),
        ruta_salida=resolver_ruta(args.output_dir),
        tecnicas=args.techniques,
        overwrite=args.overwrite,
        random_state=args.random_state,
        datasets_permitidos=datasets_permitidos,
    )


if __name__ == "__main__":
    main()

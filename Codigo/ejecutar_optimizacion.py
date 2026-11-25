"""
ejecutar_optimizacion.py
------------------------
Corre DA, PSO y DE sobre un problema (svm_lineal, svm_rbf o rf) y guarda las
curvas de convergencia y los mejores hiperparámetros en resultados/<problema>.json

Uso:
    python ejecutar_optimizacion.py svm_lineal
    python ejecutar_optimizacion.py svm_rbf
    python ejecutar_optimizacion.py rf
    python ejecutar_optimizacion.py svm_rbf --semillas 1 2 3 4 5   # corridas extra
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from metaheuristicas import ALGORITMOS
from datos_y_objetivos import ESPACIOS, Fitness, cargar_datos, decodificar

# Presupuesto idéntico para los tres algoritmos: n_pop * (max_iter + 1) evaluaciones
CONFIG = {
    "svm_lineal": {"n_pop": 20, "max_iter": 40},
    "svm_rbf":    {"n_pop": 20, "max_iter": 40},
    "rf":         {"n_pop": 15, "max_iter": 30},
}

SALIDA = Path(__file__).with_name("resultados")
SALIDA.mkdir(exist_ok=True)


def correr(problema, semilla):
    bipolar = problema.startswith("svm")
    Xtr, Xte, ytr, yte = cargar_datos(bipolar=bipolar)
    esp, cfg = ESPACIOS[problema], CONFIG[problema]
    salida = {}
    for nombre, alg in ALGORITMOS.items():
        fit = Fitness(problema, Xtr, ytr)
        t0 = time.time()
        r = alg(fit, esp["lb"], esp["ub"], seed=semilla, **cfg)
        dt = time.time() - t0
        salida[nombre] = {
            "best_x": r["best_x"].tolist(),
            "best_f": float(r["best_f"]),
            "params": decodificar(problema, r["best_x"]),
            "curva": r["curva"].tolist(),
            "n_eval": r["n_eval"],
            "evaluaciones_unicas": len(fit.cache),
            "tiempo_s": dt,
        }
        print(f"{problema:10s} semilla={semilla:3d} {nombre:4s} fitness={r['best_f']:.5f} "
              f"acc_cv={1 - r['best_f']:.4f} params={salida[nombre]['params']} ({dt:.0f}s)", flush=True)
    return salida


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("problema", choices=list(CONFIG))
    ap.add_argument("--semillas", type=int, nargs="+", default=[42])
    args = ap.parse_args()

    for s in args.semillas:
        res = correr(args.problema, s)
        nombre = f"{args.problema}.json" if s == 42 else f"{args.problema}_s{s}.json"
        (SALIDA / nombre).write_text(json.dumps({"config": CONFIG[args.problema],
                                                 "semilla": s, "resultados": res}, indent=2))

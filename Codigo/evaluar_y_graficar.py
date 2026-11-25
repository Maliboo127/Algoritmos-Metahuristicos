"""
evaluar_y_graficar.py
---------------------
Lee resultados/*.json, entrena cada modelo con los hiperparámetros encontrados
sobre el 80 % de entrenamiento y lo evalúa en el 20 % de prueba.

Genera en resultados/:
    convergencia_<problema>.png     curvas de convergencia de DA, PSO y DE
    matrices_<problema>.png         matrices de confusión (prueba)
    metricas.csv / metricas.md      exactitud, precisión, sensibilidad, especificidad, F1
    robustez.csv                    estadísticas de corridas extra (si existen)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
import numpy as np

RCV = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=7)


def cv_repetida(modelo, bipolar):
    """Exactitud con CV 10x5 sobre las 768 muestras y particiones distintas a las de la optimización."""
    Xtr, Xte, ytr, yte = cargar_datos(bipolar=bipolar)
    X, y = np.vstack([Xtr, Xte]), np.concatenate([ytr, yte])
    a = cross_val_score(modelo, X, y, cv=RCV, scoring='accuracy')
    return a.mean(), a.std()

from datos_y_objetivos import cargar_datos, construir_modelo, CV, SEMILLA

RES = Path(__file__).with_name("resultados")
TITULOS = {"svm_lineal": "SVM con kernel lineal", "svm_rbf": "SVM con kernel RBF", "rf": "Random Forest"}
COLORES = {"DA": "#2a78d6", "PSO": "#eb6834", "DE": "#1baf7a"}   # paleta categórica validada (orden fijo)
ESTILOS = {"DA": "-", "PSO": "--", "DE": "-."}                  # codificación secundaria (no solo color)
TINTA, TINTA2, REJILLA, EJE = "#1f1f1d", "#5f5e58", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.edgecolor": EJE, "axes.labelcolor": TINTA, "text.color": TINTA,
    "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.grid": True, "grid.color": REJILLA,
    "grid.linewidth": 0.6, "mathtext.fontset": "stix", "axes.spines.top": False, "axes.spines.right": False,
})


def metricas(y_true, y_pred, positivo):
    cm = confusion_matrix(y_true, y_pred, labels=[-positivo if positivo == 1 and -1 in y_true else 0, positivo])
    tn, fp, fn, tp = cm.ravel()
    return {
        "exactitud": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=positivo),
        "sensibilidad": recall_score(y_true, y_pred, pos_label=positivo),
        "especificidad": tn / (tn + fp),
        "f1": f1_score(y_true, y_pred, pos_label=positivo),
        "cm": cm,
    }


def linea_base(problema):
    """Modelo con los hiperparámetros por defecto de scikit-learn (referencia)."""
    if problema == "svm_lineal":
        clf = SVC(kernel="linear")
    elif problema == "svm_rbf":
        clf = SVC(kernel="rbf")            # C = 1, gamma = 'scale'
    else:
        clf = RandomForestClassifier(random_state=SEMILLA, n_jobs=-1)
    pasos = [("imp", SimpleImputer(strategy="median"))]
    if problema.startswith("svm"):
        pasos.append(("esc", StandardScaler()))
    return Pipeline(pasos + [("clf", clf)])


def graficar_convergencia(problema, res):
    fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=300)
    for alg, r in res.items():
        curva = np.array(r["curva"])
        ax.plot(np.arange(len(curva)), curva, ESTILOS[alg], color=COLORES[alg], lw=1.6, label=alg)
    ax.set_xlabel("Iteración")
    ax.set_ylabel(r"fitness $= 1-\overline{Acc}_{CV}$")
    ax.set_title(TITULOS[problema], fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(RES / f"convergencia_{problema}.png")
    plt.close(fig)


def graficar_matrices(problema, cms):
    fig, axes = plt.subplots(1, len(cms), figsize=(3.5, 1.55), dpi=300)
    etiquetas = ["No diab.", "Diab."]
    for ax, (alg, cm) in zip(axes, cms.items()):
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum())
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > cm.sum() * 0.45 else TINTA)
        ax.set_title(alg, fontsize=8)
        ax.set_xticks([0, 1], etiquetas, fontsize=6)
        ax.set_yticks([0, 1], etiquetas if ax is axes[0] else ["", ""], fontsize=6)
        ax.grid(False); ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("Real", fontsize=7)
    for ax in axes:
        ax.set_xlabel("Predicho", fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(RES / f"matrices_{problema}.png")
    plt.close(fig)


def main():
    filas = []
    for problema in ["svm_lineal", "svm_rbf", "rf"]:
        ruta = RES / f"{problema}.json"
        if not ruta.exists():
            continue
        res = json.loads(ruta.read_text())["resultados"]
        bipolar = problema.startswith("svm")
        positivo = 1
        Xtr, Xte, ytr, yte = cargar_datos(bipolar=bipolar)

        graficar_convergencia(problema, res)
        cms = {}
        # Línea base
        base = linea_base(problema).fit(Xtr, ytr)
        m = metricas(yte, base.predict(Xte), positivo)
        acc_cv = cross_val_score(linea_base(problema), Xtr, ytr, cv=CV).mean()
        rm, rs = cv_repetida(linea_base(problema), bipolar)
        filas.append({"problema": problema, "algoritmo": "Por defecto", "params": "default sklearn",
                      "acc_cv": acc_cv, "rcv_media": rm, "rcv_std": rs, **{k: v for k, v in m.items() if k != "cm"},
                      "cm": m["cm"].tolist(), "iter_mejor": None})
        for alg, r in res.items():
            modelo = construir_modelo(problema, r["params"]).fit(Xtr, ytr)
            m = metricas(yte, modelo.predict(Xte), positivo)
            cms[alg] = m["cm"]
            curva = np.array(r["curva"])
            rm, rs = cv_repetida(construir_modelo(problema, r["params"]), bipolar)
            filas.append({"problema": problema, "algoritmo": alg, "params": r["params"],
                          "acc_cv": 1 - r["best_f"], "rcv_media": rm, "rcv_std": rs, **{k: v for k, v in m.items() if k != "cm"},
                          "cm": m["cm"].tolist(),
                          "iter_mejor": int(np.argmax(curva <= curva[-1] + 1e-12)),
                          "tiempo_s": r["tiempo_s"], "unicas": r["evaluaciones_unicas"]})
        graficar_matrices(problema, cms)

    df = pd.DataFrame(filas)
    df.to_csv(RES / "metricas.csv", index=False)
    with open(RES / "metricas.md", "w") as fh:
        fh.write(df.drop(columns=["cm"]).to_markdown(index=False, floatfmt=".4f"))
    print(df.drop(columns=["cm"]).to_string())
    for _, f in df.iterrows():
        print(f["problema"], f["algoritmo"], f["cm"])

    # Robustez: corridas con semillas extra
    filas_r = []
    for ruta in sorted(RES.glob("*_s*.json")):
        d = json.loads(ruta.read_text())
        problema = ruta.stem.rsplit("_s", 1)[0]
        for alg, r in d["resultados"].items():
            filas_r.append({"problema": problema, "semilla": d["semilla"], "algoritmo": alg,
                            "acc_cv": 1 - r["best_f"],
                            "iter_mejor": int(np.argmax(np.array(r["curva"]) <= r["curva"][-1] + 1e-12))})
    # incluir la corrida principal (semilla 42)
    for problema in ["svm_lineal", "svm_rbf", "rf"]:
        ruta = RES / f"{problema}.json"
        if ruta.exists() and any(f["problema"] == problema for f in filas_r):
            for alg, r in json.loads(ruta.read_text())["resultados"].items():
                filas_r.append({"problema": problema, "semilla": 42, "algoritmo": alg,
                                "acc_cv": 1 - r["best_f"],
                                "iter_mejor": int(np.argmax(np.array(r["curva"]) <= r["curva"][-1] + 1e-12))})
    if filas_r:
        dr = pd.DataFrame(filas_r)
        resumen = dr.groupby(["problema", "algoritmo"]).agg(
            corridas=("acc_cv", "size"), acc_media=("acc_cv", "mean"), acc_std=("acc_cv", "std"),
            acc_max=("acc_cv", "max"), acc_min=("acc_cv", "min"), iter_media=("iter_mejor", "mean"))
        resumen.to_csv(RES / "robustez.csv")
        print(resumen)


if __name__ == "__main__":
    main()

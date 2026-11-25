"""
datos_y_objetivos.py
--------------------
Carga del Pima Indians Diabetes dataset, preprocesamiento y definición de las
funciones objetivo (fitness) para la SVM y el Random Forest.

    fitness(x) = 1 - (1/K) Σ_k Accuracy_k(x),   K = 5  (validación cruzada estratificada)

Decisiones de preprocesamiento
    * En Glucose, BloodPressure, SkinThickness, Insulin y BMI un valor 0 es
      fisiológicamente imposible: se trata como dato faltante (NaN) y se imputa
      con la mediana. La imputación y el escalado van dentro de un Pipeline de
      scikit-learn, así se ajustan solo con los lotes de entrenamiento de cada
      fold (sin fuga de información).
    * Se reserva un 20 % estratificado como conjunto de prueba. La optimización
      solo ve el 80 % de entrenamiento; las métricas finales se miden en prueba.
    * Para la SVM las etiquetas se convierten a formato bipolar {-1, +1}.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

SEMILLA = 42
K_FOLDS = 5
COLS_CERO_FALTANTE = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
RUTA_CSV = Path(__file__).with_name("diabetes_dataset.csv")


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def cargar_datos(ruta=RUTA_CSV, bipolar=False):
    df = pd.read_csv(ruta)
    df[COLS_CERO_FALTANTE] = df[COLS_CERO_FALTANTE].replace(0, np.nan)
    X = df.drop(columns="Outcome").values
    y = df["Outcome"].values.astype(int)
    if bipolar:
        y = np.where(y == 1, 1, -1)          # (0, 1) -> (-1, +1)
    return train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEMILLA)


CV = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEMILLA)


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
def modelo_svm(kernel, C, gamma=None):
    params = dict(kernel=kernel, C=C)
    if kernel == "rbf":
        params["gamma"] = gamma
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("esc", StandardScaler()),
        ("svm", SVC(**params)),
    ])


def modelo_rf(n_estimators, max_depth, min_samples_split, min_samples_leaf, n_jobs=-1):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=int(n_estimators), max_depth=int(max_depth),
            min_samples_split=int(min_samples_split), min_samples_leaf=int(min_samples_leaf),
            random_state=SEMILLA, n_jobs=n_jobs)),
    ])


# ---------------------------------------------------------------------------
# Espacios de búsqueda y decodificación de los vectores de decisión
# ---------------------------------------------------------------------------
# SVM. C ∈ [0.1, 100] y γ ∈ [1e-4, 10]. Los algoritmos buscan sobre log10 de
# cada parámetro (mismo intervalo, pero recorrido en escala logarítmica, que es
# la natural para estos hiperparámetros: 0.1→1 pesa lo mismo que 10→100).
#   kernel lineal : x = [C]         (γ no interviene en el kernel lineal)
#   kernel RBF    : x = [C, γ]
ESPACIOS = {
    "svm_lineal": {"lb": np.log10([0.1]),         "ub": np.log10([100.0]),
                   "nombres": ["C"]},
    "svm_rbf":    {"lb": np.log10([0.1, 1e-4]),   "ub": np.log10([100.0, 10.0]),
                   "nombres": ["C", "gamma"]},
    # RF: variables enteras. Se busca en el continuo [lb-0.5, ub+0.5] y se redondea,
    # para que los extremos tengan la misma probabilidad que los valores interiores.
    "rf": {"lb": np.array([10, 3, 2, 1]) - 0.499, "ub": np.array([300, 30, 20, 20]) + 0.499,
           "nombres": ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]},
}


def decodificar(problema, x):
    x = np.asarray(x, float)
    if problema == "svm_lineal":
        return {"C": float(10 ** x[0])}
    if problema == "svm_rbf":
        return {"C": float(10 ** x[0]), "gamma": float(10 ** x[1])}
    if problema == "rf":
        lo, hi = np.array([10, 3, 2, 1]), np.array([300, 30, 20, 20])
        v = np.clip(np.rint(x).astype(int), lo, hi)
        return dict(zip(ESPACIOS["rf"]["nombres"], map(int, v)))
    raise ValueError(problema)


def construir_modelo(problema, params):
    if problema == "svm_lineal":
        return modelo_svm("linear", params["C"])
    if problema == "svm_rbf":
        return modelo_svm("rbf", params["C"], params["gamma"])
    if problema == "rf":
        return modelo_rf(**params)
    raise ValueError(problema)


class Fitness:
    """
    Función objetivo con memoria: fitness(x) = 1 - accuracy media en CV de 5 lotes.
    La caché evita reentrenar cuando dos individuos se decodifican en los mismos
    hiperparámetros (frecuente en el RF, que es entero). Las evaluaciones se
    siguen contando para el presupuesto aunque vengan de la caché.
    """

    def __init__(self, problema, X, y):
        self.problema, self.X, self.y = problema, X, y
        self.cache = {}
        self.llamadas = 0

    def __call__(self, x):
        self.llamadas += 1
        params = decodificar(self.problema, x)
        clave = tuple(round(v, 10) if isinstance(v, float) else v for v in params.values())
        if clave not in self.cache:
            acc = cross_val_score(construir_modelo(self.problema, params),
                                  self.X, self.y, cv=CV, scoring="accuracy")
            self.cache[clave] = 1.0 - acc.mean()
        return self.cache[clave]

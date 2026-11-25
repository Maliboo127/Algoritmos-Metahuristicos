"""Genera el reporte IEEE (docx) a partir de resultados/metricas.csv y resultados/robustez.csv."""
import ast
import json
from pathlib import Path

import pandas as pd
from docxkit import Reporte

RAIZ = Path(__file__).resolve().parent.parent
RES = RAIZ / "resultados"
PLANTILLA = Path("/home/claude/tpl/plantilla.docx")
SALIDA = RAIZ / "reporte" / "Reporte_IEEE_Proyecto_Final_Metaheuristicos.docx"

met = pd.read_csv(RES / "metricas.csv")
rob = pd.read_csv(RES / "robustez.csv")
M = {(r.problema, r.algoritmo): r for r in met.itertuples()}
RB = {(r.problema, r.algoritmo): r for r in rob.itertuples()}
CFG = {p: json.loads((RES / f"{p}.json").read_text())["config"] for p in ["svm_lineal", "svm_rbf", "rf"]}
ALGS = ["DA", "PSO", "DE"]


def pct(x, d=2):
    return f"{100 * x:.{d}f}".replace(".", ".")


def pr(p, a):
    v = M[(p, a)].params
    return ast.literal_eval(v) if v.startswith("{") else v


def cm(p, a):
    return ast.literal_eval(M[(p, a)].cm)


R = Reporte(PLANTILLA)
R.titulo("Optimización de hiperparámetros de SVM y Random Forest con los algoritmos Dragonfly, PSO y Evolución Diferencial para el Pima Indians Diabetes Dataset")
afil = ["CUCEI", "Universidad de Guadalajara", "Guadalajara, México"]
R.autores([
    ["Marlene Ximena Sandoval García", *afil, "Código: 219511456"],
    ["Yael Iván García Mercado", *afil],
    ["Marcos Daniel Díaz Briones", *afil],
    ["Leonardo Daniel Palacios Palacios", *afil],
    ["Oscar Javier Cervantes Cervera", *afil],
])
R.pie_de_pagina("Algoritmos Metaheurísticos · Sección D02 · Prof. Mario Antonio Ruz Canul · CUCEI, UdG · 2026")

# ---------------------------------------------------------------- resumen
best_rbf = M[("svm_rbf", "DE")]
R.abstract(
    "Este trabajo compara tres algoritmos metaheurísticos, el Dragonfly Algorithm (DA), la optimización por enjambre de "
    "partículas (PSO) y la evolución diferencial (DE), para ajustar los hiperparámetros de una máquina de soporte vectorial "
    "(SVM) con kernel lineal y RBF y de un Random Forest (RF) sobre el Pima Indians Diabetes Dataset. Los tres algoritmos "
    "se programaron desde cero y minimizan el error de validación cruzada estratificada de cinco lotes, "
    "fitness(x) = 1 − exactitud media. Con el mismo presupuesto de evaluaciones, los tres alcanzan la misma exactitud de "
    f"validación en la SVM-RBF ({pct(best_rbf.acc_cv)} %) y en la SVM lineal ({pct(M[('svm_lineal','DA')].acc_cv)} %); en el RF, "
    f"PSO y DE ({pct(M[('rf','PSO')].acc_cv)} %) superan a DA ({pct(M[('rf','DA')].acc_cv)} %). En corridas repetidas con "
    "distintas semillas, DE alcanzó el mejor valor de la SVM-RBF en las 10 corridas y DA fue el que más se estancó. La SVM-RBF "
    "fue el mejor modelo tanto en la función objetivo como en una validación cruzada repetida 10×5 independiente, aunque en "
    "el conjunto de prueba el RF ajustado obtuvo mayor exactitud y sensibilidad. "
    "Las diferencias entre modelos son de uno a tres puntos porcentuales y quedan dentro de la variabilidad estadística "
    "que permite un conjunto de 768 muestras."
)
R.keywords("algoritmos metaheurísticos, Dragonfly Algorithm, PSO, evolución diferencial, optimización de hiperparámetros, SVM, random forest, diabetes")

# ---------------------------------------------------------------- I. Introducción
R.h1("Introducción")
R.par("El desempeño de un modelo de aprendizaje automático depende en gran medida de sus hiperparámetros, es decir, de los "
      "valores que se fijan antes del entrenamiento y que el propio algoritmo de aprendizaje no ajusta. En una SVM, la "
      "constante de regularización *C* y el ancho del kernel *γ* controlan el compromiso entre margen y error de entrenamiento; "
      "en un Random Forest, el número de árboles y las reglas de crecimiento de cada árbol controlan la varianza del ensamble. "
      "Elegir mal estos valores produce modelos subajustados o sobreajustados.")
R.par("La búsqueda en malla y la búsqueda aleatoria son los métodos más comunes, pero su costo crece rápidamente con la "
      "dimensión y la resolución del espacio de búsqueda. La función que relaciona hiperparámetros con error de validación no es "
      "diferenciable, es ruidosa y presenta mesetas, por lo que no admite métodos basados en gradiente. Los algoritmos "
      "metaheurísticos poblacionales, inspirados en procesos evolutivos o en el comportamiento colectivo de animales, son una "
      "alternativa natural para este tipo de problemas de caja negra [1]–[4].")
R.par("El objetivo de este proyecto es diseñar, implementar y evaluar tres metaheurísticas para encontrar la configuración de "
      "hiperparámetros que minimiza el error de generalización estimado por validación cruzada. Se empleó el Dragonfly "
      "Algorithm (DA) [1], algoritmo que el equipo presentó en la exposición, junto con PSO [2], [3] y la evolución "
      "diferencial DE/rand/1/bin [4]. Los tres se programaron desde cero en Python con NumPy; scikit-learn [5] se usa solo para "
      "los modelos de clasificación y la validación cruzada. Se resuelven dos casos sobre el Pima Indians Diabetes Dataset [6]: "
      "una SVM [7] con kernel lineal y RBF y un Random Forest [8] con cuatro hiperparámetros enteros.")

# ---------------------------------------------------------------- II. Datos
R.h1("Conjunto de datos y preprocesamiento")
R.par("El Pima Indians Diabetes Dataset [6] contiene 768 registros de mujeres de al menos 21 años de herencia pima, con ocho "
      "variables clínicas (embarazos, glucosa, presión arterial, grosor del pliegue cutáneo, insulina, IMC, función de "
      "pedigrí de diabetes y edad) y una etiqueta binaria que indica diagnóstico de diabetes (500 negativos y 268 positivos, "
      "34.9 % de casos positivos).")
R.par("En las columnas de glucosa, presión arterial, pliegue cutáneo, insulina e IMC un valor de cero es fisiológicamente "
      "imposible y representa un dato faltante (5, 35, 227, 374 y 11 registros, respectivamente). Estos ceros se reemplazaron "
      "por valores faltantes y se imputaron con la mediana. Para la SVM, además, las variables se estandarizaron a media cero y "
      "varianza unitaria. Ambas transformaciones forman parte de un *Pipeline* de scikit-learn, de modo que en cada lote de la "
      "validación cruzada se ajustan únicamente con los datos de entrenamiento de ese lote y no hay fuga de información.")
R.par("Como pide el enunciado, para la SVM las etiquetas se transformaron del formato (0, 1) al formato bipolar (−1, +1), donde "
      "+1 corresponde a la clase diabética. Los datos se dividieron de forma estratificada en 80 % de entrenamiento (614 "
      "registros) y 20 % de prueba (154 registros: 100 negativos y 54 positivos). La optimización solo usa el conjunto de "
      "entrenamiento; el de prueba se reserva para medir las métricas finales.")

# ---------------------------------------------------------------- III. Metodología
R.h1("Metodología")
R.h2("Función objetivo")
R.par("Cada vector de decisión *x* se evalúa con validación cruzada estratificada de *K* = 5 lotes sobre el conjunto de "
      "entrenamiento, y se minimiza")
R.ecuacion("fitness(*x*) = 1 − (1/*K*) Σ_{k=1}^{K} Accuracy_{k}(*x*)", 1)
R.par("Las particiones de la validación cruzada se fijaron con una semilla, de modo que la función objetivo es determinista y "
      "los tres algoritmos optimizan exactamente la misma función. Como cada lote tiene 122 o 123 muestras, la exactitud "
      "solo cambia en saltos de aproximadamente 0.16 %, por lo que la superficie de búsqueda está formada por mesetas. "
      "Para no reentrenar modelos idénticos, la función guarda en caché el resultado de cada combinación de hiperparámetros; "
      "las evaluaciones en caché se cuentan igualmente para el presupuesto.")

R.h2("Vectores de decisión y espacios de búsqueda")
R.par("El vector de decisión depende del modelo. En la SVM con kernel lineal, *K*(*u*, *v*) = *u*^{T}*v* no depende de *γ*, de modo "
      "que el vector se reduce a *x* = [*C*]. Con kernel RBF, *K*(*u*, *v*) = exp(−*γ*‖*u* − *v*‖²), y el vector es *x* = [*C*, *γ*]. "
      "Los intervalos son los del enunciado, *C* ∈ [0.1, 100] y *γ* ∈ [10^{−4}, 10]. Como estos intervalos abarcan tres y cinco "
      "órdenes de magnitud, los algoritmos los recorren en escala logarítmica, es decir, buscan log_{10}*C* ∈ [−1, 2] y "
      "log_{10}*γ* ∈ [−4, 1] y decodifican *C* = 10^{x1} y *γ* = 10^{x2}. El espacio es el mismo, pero así el intervalo "
      "[0.1, 1] recibe la misma atención que [10, 100].")
R.par("En el Random Forest, *x* = [n_estimators, max_depth, min_samples_split, min_samples_leaf], con los cuatro valores "
      "enteros en [10, 300], [3, 30], [2, 20] y [1, 20]. Los algoritmos trabajan en el continuo [*l* − 0.5, *u* + 0.5] y cada "
      "componente se redondea al entero más cercano antes de evaluar, lo que da a los extremos la misma probabilidad que a los "
      "valores interiores. La Tabla I resume ambos espacios.")
R.tabla("Vectores de decisión y espacios de búsqueda",
        ["Modelo", "Vector *x*", "Intervalos", "Tipo / codificación"],
        [["SVM lineal", "[*C*]", "*C* ∈ [0.1, 100]", "Real, log_{10}"],
         ["SVM RBF", "[*C*, *γ*]", "*C* ∈ [0.1, 100], *γ* ∈ [10^{−4}, 10]", "Real, log_{10}"],
         ["Random Forest", "[*n*_{est}, *d*_{max}, *s*_{min}, *l*_{min}]", "[10,300], [3,30], [2,20], [1,20]", "Entero (redondeo)"]],
        [0.75, 0.95, 1.45, 0.85])

R.h2("Dragonfly Algorithm (DA)")
R.par("El DA [1] imita el enjambre estático (caza) y dinámico (migración) de las libélulas. Cada libélula *i* tiene una "
      "posición *X*_{i} y un vector de paso Δ*X*_{i}. Respecto a sus vecinos *j* = 1,…,*N* dentro de un radio *r* se calculan "
      "cinco comportamientos: separación *S*_{i} = −Σ(*X*_{i} − *X*_{j}), alineación *A*_{i} = media(Δ*X*_{j}), cohesión "
      "*C*_{i} = media(*X*_{j}) − *X*_{i}, atracción a la comida *F*_{i} = *X*^{+} − *X*_{i} y distracción del enemigo "
      "*E*_{i} = *X*^{−} + *X*_{i}, donde *X*^{+} y *X*^{−} son la mejor y la peor solución encontradas. El paso y la posición se "
      "actualizan con")
R.ecuacion("Δ*X*_{t+1} = *sS* + *aA* + *cC* + *fF* + *eE* + *w*Δ*X*_{t}", 2)
R.ecuacion("*X*_{t+1} = *X*_{t} + Δ*X*_{t+1}", 3)
R.par("Si una libélula no tiene vecinos, realiza un vuelo de Lévy, *X*_{t+1} = *X*_{t} + Lévy(*d*)·*X*_{t}, generado con el "
      "algoritmo de Mantegna (*β* = 1.5). En nuestra implementación el paso de Lévy se escala por el ancho del intervalo de cada "
      "variable para que sea comparable entre dimensiones con rangos muy distintos. Los pesos son adaptativos, como en el "
      "artículo original: *w* decrece linealmente de 0.9 a 0.4; *s*, *a* y *c* se toman como 2·rand·*c*_{t}, *f* = 2·rand y "
      "*e* = *c*_{t}, con *c*_{t} = 0.1 − 0.2*t*/*T* (acotado en 0), y el radio de vecindad crece de un cuarto del rango hasta "
      "cubrir todo el espacio. Así, al inicio domina la exploración y al final la explotación. El paso se acota a ±10 % del rango.")

R.h2("Optimización por enjambre de partículas (PSO)")
R.par("En PSO [2], [3] cada partícula recuerda su mejor posición *p*_{i} y conoce la mejor del enjambre *g*. Se usó la variante "
      "global con peso de inercia linealmente decreciente:")
R.ecuacion("*v*_{t+1} = *wv*_{t} + *c*_{1}*r*_{1}(*p*_{i} − *x*_{t}) + *c*_{2}*r*_{2}(*g* − *x*_{t})", 4)
R.ecuacion("*x*_{t+1} = *x*_{t} + *v*_{t+1}", 5)
R.par("con *w* de 0.9 a 0.4, *c*_{1} = *c*_{2} = 2, *r*_{1}, *r*_{2} ~ U(0, 1) y la velocidad limitada a ±20 % del rango. Las posiciones "
      "que salen del espacio se proyectan al límite más cercano.")

R.h2("Evolución diferencial (DE/rand/1/bin)")
R.par("DE [4] es un algoritmo evolutivo que genera un vector mutante a partir de tres individuos distintos elegidos al azar, "
      "*v* = *x*_{r1} + *F*(*x*_{r2} − *x*_{r3}). Después, la cruza binomial forma el vector de prueba *u* tomando cada "
      "componente de *v* con probabilidad *CR* (al menos una componente siempre viene de *v*). Por último, la selección "
      "codiciosa reemplaza a *x*_{i} por *u* si *f*(*u*) ≤ *f*(*x*_{i}). Se usaron *F* = 0.5 y *CR* = 0.9.")

R.h2("Configuración experimental")
R.par("Para que la comparación sea justa, los tres algoritmos usan la misma población inicial (misma semilla), el mismo tamaño "
      "de población y el mismo número de iteraciones, es decir, el mismo número de evaluaciones de la función objetivo "
      "(Tabla II). La corrida principal usa la semilla 42. Para medir la robustez se repitió la optimización con semillas "
      "adicionales (10 corridas en total para la SVM-RBF, 5 para la SVM lineal y "
      f"{int(RB[('rf','DA')].corridas)} para el RF).")
n_ev = {p: CFG[p]["n_pop"] * (CFG[p]["max_iter"] + 1) for p in CFG}
R.tabla("Parámetros de los algoritmos",
        ["Parámetro", "SVM (lineal y RBF)", "Random Forest"],
        [["Población / iteraciones", f"{CFG['svm_rbf']['n_pop']} / {CFG['svm_rbf']['max_iter']}", f"{CFG['rf']['n_pop']} / {CFG['rf']['max_iter']}"],
         ["Evaluaciones por corrida", str(n_ev["svm_rbf"]), str(n_ev["rf"])],
         ["DA", "*w*: 0.9→0.4, *c*_{t}: 0.1→0, Lévy *β* = 1.5", "igual"],
         ["PSO", "*w*: 0.9→0.4, *c*_{1} = *c*_{2} = 2, *v*_{max} = 0.2·rango", "igual"],
         ["DE", "DE/rand/1/bin, *F* = 0.5, *CR* = 0.9", "igual"]],
        [1.15, 1.6, 0.85])
R.par("Con los mejores hiperparámetros de cada algoritmo se entrenó el modelo sobre las 614 muestras de entrenamiento y se "
      "evaluó sobre las 154 de prueba. Tomando la clase diabética como positiva, se reportan exactitud (TP+TN)/N, "
      "precisión TP/(TP+FP), sensibilidad TP/(TP+FN), especificidad TN/(TN+FP), F1 y la matriz de confusión. Como referencia "
      "se incluye cada modelo con los hiperparámetros por defecto de scikit-learn (SVM: *C* = 1, *γ* = 'scale'; RF: 100 árboles "
      "sin límite de profundidad). Además, como el conjunto de prueba es pequeño (una muestra equivale a 0.65 %), cada "
      "configuración final se evaluó con validación cruzada estratificada repetida 10×5 sobre las 768 muestras, con "
      "particiones distintas a las usadas en la optimización.")

# ---------------------------------------------------------------- IV. Resultados
R.h1("Resultados")
R.h2("SVM con kernel lineal y RBF")
R.par("Las Figs. 1 y 2 muestran las curvas de convergencia (mejor fitness acumulado por iteración) de la corrida principal. En el "
      "kernel lineal el problema es unidimensional y la exactitud de validación es prácticamente constante para *C* ≳ 1, por lo "
      f"que los tres algoritmos ya tenían el óptimo de {pct(M[('svm_lineal','DA')].acc_cv)} % en la población inicial (iteración 0) "
      "y la curva es plana. En el kernel RBF los tres algoritmos parten de la misma población y convergen al mismo valor, "
      f"fitness = {1 - best_rbf.acc_cv:.4f} (exactitud de validación de {pct(best_rbf.acc_cv)} %). DE lo alcanzó en la iteración "
      f"{int(M[('svm_rbf','DE')].iter_mejor)}, PSO en la {int(M[('svm_rbf','PSO')].iter_mejor)} y DA en la "
      f"{int(M[('svm_rbf','DA')].iter_mejor)}.")
R.figura(RES / "convergencia_svm_lineal.png", "Curvas de convergencia de DA, PSO y DE para la SVM con kernel lineal (semilla 42).", 3.2)
R.figura(RES / "convergencia_svm_rbf.png", "Curvas de convergencia de DA, PSO y DE para la SVM con kernel RBF (semilla 42).", 3.2)
# La plantilla numera las figuras automáticamente; el texto se refiere a ellas como Fig. 1 y Fig. 2.

filas = []
for p, etq in [("svm_lineal", "Lineal"), ("svm_rbf", "RBF")]:
    for a in ALGS:
        prm = pr(p, a)
        g = f"{prm['gamma']:.2e}".replace("e-0", "×10^{−").replace("e-", "×10^{−") + "}" if "gamma" in prm else "—"
        filas.append([etq, a, f"{prm['C']:.3f}", g, pct(M[(p, a)].acc_cv)])
R.tabla("Hiperparámetros óptimos de la SVM (semilla 42)",
        ["Kernel", "Alg.", "*C*", "*γ*", "Acc. CV (%)"], filas, [0.6, 0.45, 0.7, 0.95, 0.75])

R.par("La Tabla III muestra que en el kernel RBF los tres algoritmos encontraron puntos distintos de una misma región: *C* "
      "alto (50–100) con *γ* ≈ 10^{−3}. Con *γ* tan pequeño, el kernel RBF se comporta casi como un kernel lineal muy suave, lo "
      "que concuerda con que el kernel lineal también alcance casi la misma exactitud. Las corridas con otras semillas "
      "(Sección IV-C) revelaron una segunda región con el mismo fitness, *C* ≈ 0.8 y *γ* ≈ 0.06, que corresponde a un "
      "modelo más regularizado y más no lineal. Es decir, la función objetivo tiene varios óptimos equivalentes.")

filas = []
for p, etq in [("svm_lineal", "Lineal"), ("svm_rbf", "RBF")]:
    for a in ["Por defecto"] + ALGS:
        m = M[(p, a)]
        filas.append([etq, "Def." if a == "Por defecto" else a, pct(m.exactitud), pct(m.precision), pct(m.sensibilidad),
                      pct(m.especificidad), pct(m.f1)])
R.tabla("Métricas de la SVM en el conjunto de prueba (%)",
        ["Kernel", "Alg.", "Exact.", "Prec.", "Sens.", "Espec.", "F1"], filas,
        [0.55, 0.45, 0.5, 0.5, 0.5, 0.5, 0.45],
        nota="Def.: hiperparámetros por defecto de scikit-learn. Clase positiva: diabética (+1).")

R.par("Las métricas de prueba (Tabla IV y Figs. 3 y 4) son casi idénticas entre algoritmos. En el kernel lineal los tres "
      f"producen las mismas predicciones que el modelo con *C* = 1: exactitud de {pct(M[('svm_lineal','DA')].exactitud)} %, "
      f"precisión de {pct(M[('svm_lineal','DA')].precision)} % y sensibilidad de {pct(M[('svm_lineal','DA')].sensibilidad)} %. "
      f"En el kernel RBF, PSO y DE obtienen {pct(M[('svm_rbf','PSO')].exactitud)} % de exactitud (un falso positivo menos que DA). "
      "La sensibilidad, de 48.15 %, es baja en todos los casos: el modelo detecta menos de la mitad de las pacientes diabéticas, "
      "un efecto del desbalance de clases y de que la función objetivo optimiza la exactitud y no la sensibilidad.")
R.figura(RES / "matrices_svm_lineal.png", "Matrices de confusión en prueba de la SVM lineal con los hiperparámetros de DA, PSO y DE.", 3.3)
R.figura(RES / "matrices_svm_rbf.png", "Matrices de confusión en prueba de la SVM RBF con los hiperparámetros de DA, PSO y DE.", 3.3)

# ---------------------------------------------------------------- RF
R.h2("Random Forest")
R.par("La Fig. 5 muestra la convergencia para el RF. PSO encontró su mejor configuración en la iteración "
      f"{int(M[('rf','PSO')].iter_mejor)} y DE en la {int(M[('rf','DE')].iter_mejor)}, ambas con fitness = "
      f"{1 - M[('rf','PSO')].acc_cv:.4f} ({pct(M[('rf','PSO')].acc_cv)} % de exactitud de validación). DA mejoró en la "
      f"iteración {int(M[('rf','DA')].iter_mejor)} y después se estancó en {1 - M[('rf','DA')].acc_cv:.4f} "
      f"({pct(M[('rf','DA')].acc_cv)} %). En este problema DA necesitó menos tiempo de cómputo "
      f"({M[('rf','DA')].tiempo_s:.0f} s frente a {M[('rf','PSO')].tiempo_s:.0f} s de PSO y {M[('rf','DE')].tiempo_s:.0f} s de DE) "
      f"porque su población se concentró pronto en pocas configuraciones: solo {int(M[('rf','DA')].unicas)} de las {n_ev['rf']} "
      "evaluaciones fueron combinaciones nuevas. Es otra señal de que perdió diversidad.")
R.figura(RES / "convergencia_rf.png", "Curvas de convergencia de DA, PSO y DE para el Random Forest (semilla 42).", 3.2)

filas = []
for a in ALGS:
    prm = pr("rf", a)
    filas.append([a, prm["n_estimators"], prm["max_depth"], prm["min_samples_split"], prm["min_samples_leaf"], pct(M[("rf", a)].acc_cv)])
R.tabla("Hiperparámetros óptimos del Random Forest (semilla 42)",
        ["Alg.", "n_estim.", "max_depth", "min_split", "min_leaf", "Acc. CV (%)"], filas,
        [0.4, 0.6, 0.75, 0.68, 0.62, 0.65])
R.par("En la Tabla V, los tres algoritmos llevaron min_samples_leaf al límite superior del intervalo (20 muestras por hoja). "
      "Es decir, en este dataset conviene regularizar fuertemente cada árbol, y es posible que el óptimo esté fuera del espacio "
      "de búsqueda definido. PSO y DE coincidieron en 165 árboles; la profundidad máxima y min_samples_split resultaron poco "
      "relevantes, porque con hojas de 20 muestras los árboles rara vez alcanzan la profundidad máxima.")

filas = []
for a in ["Por defecto"] + ALGS:
    m = M[("rf", a)]
    filas.append(["Def." if a == "Por defecto" else a, pct(m.exactitud), pct(m.precision), pct(m.sensibilidad), pct(m.especificidad), pct(m.f1)])
R.tabla("Métricas del Random Forest en el conjunto de prueba (%)",
        ["Alg.", "Exact.", "Prec.", "Sens.", "Espec.", "F1"], filas, [0.6, 0.55, 0.55, 0.55, 0.55, 0.5])
R.par(f"Con los hiperparámetros de PSO y DE, el RF obtuvo {pct(M[('rf','PSO')].exactitud)} % de exactitud, "
      f"{pct(M[('rf','PSO')].precision)} % de precisión, {pct(M[('rf','PSO')].sensibilidad)} % de sensibilidad y "
      f"{pct(M[('rf','PSO')].especificidad)} % de especificidad (Tabla VI y Fig. 6), mejor que DA en todas las métricas. "
      "Estos son los mejores valores de prueba entre los modelos optimizados.")
R.figura(RES / "matrices_rf.png", "Matrices de confusión en prueba del Random Forest con los hiperparámetros de DA, PSO y DE.", 3.3)

# ---------------------------------------------------------------- Robustez
R.h2("Robustez frente a la semilla aleatoria")
R.par("Una sola corrida no basta para comparar algoritmos estocásticos. La Tabla VII resume las corridas con distintas "
      "semillas: la media y la desviación estándar de la mejor exactitud de validación, y la iteración media en la que se "
      "encontró el mejor valor.")
filas = []
for p, etq in [("svm_lineal", "SVM lin."), ("svm_rbf", "SVM RBF"), ("rf", "RF")]:
    for a in ALGS:
        r = RB[(p, a)]
        filas.append([etq, a, int(r.corridas), f"{100*r.acc_media:.2f} ± {100*r.acc_std:.2f}", pct(r.acc_max), f"{r.iter_media:.1f}"])
R.tabla("Resultados de corridas independientes (exactitud CV, %)",
        ["Modelo", "Alg.", "*n*", "Media ± d.e.", "Máx.", "Iter."], filas, [0.6, 0.4, 0.3, 0.9, 0.5, 0.45])

# texto de robustez basado en los datos
def exitos(p, a, tol=5e-4):
    import glob
    best = max(RB[(p, x)].acc_max for x in ALGS)
    n = 0; tot = 0
    for f in [RES / f"{p}.json", *sorted(RES.glob(f"{p}_s*.json"))]:
        d = json.loads(Path(f).read_text())["resultados"][a]
        tot += 1; n += (1 - d["best_f"]) >= best - tol
    return n, tot

e = {a: exitos("svm_rbf", a) for a in ALGS}
er = {a: exitos("rf", a) for a in ALGS}
R.par(f"En la SVM-RBF, el mejor valor conocido ({pct(RB[('svm_rbf','DE')].acc_max)} %) lo alcanzó DE en {e['DE'][0]} de "
      f"{e['DE'][1]} corridas, PSO en {e['PSO'][0]} de {e['PSO'][1]} y DA en {e['DA'][0]} de {e['DA'][1]}. DA quedó "
      "atrapado con frecuencia en una meseta de 78.99 % con *C* ≈ 5 y *γ* ≈ 5×10^{−3}. En el RF, el mejor valor "
      f"({pct(max(RB[('rf',a)].acc_max for a in ALGS))} %) lo alcanzó DE en {er['DE'][0]} de {er['DE'][1]} corridas, PSO en "
      f"{er['PSO'][0]} de {er['PSO'][1]} y DA en {er['DA'][0]} de {er['DA'][1]}; ese valor corresponde a un bosque muy distinto "
      "al de la corrida principal (10 árboles de profundidad 4), otra muestra de que hay varios óptimos casi equivalentes. "
      f"Por medias, PSO ({100*RB[('rf','PSO')].acc_media:.2f} %) y DE ({100*RB[('rf','DE')].acc_media:.2f} %) superan a DA "
      f"({100*RB[('rf','DA')].acc_media:.2f} %). El comportamiento de DA se explica por su "
      "diseño: los pesos de separación, alineación y cohesión se anulan en la segunda mitad de la ejecución y el radio de "
      "vecindad crece hasta abarcar a todo el enjambre, de modo que en problemas con mesetas, donde la mejor solución deja "
      "de cambiar, todas las libélulas se agrupan alrededor de la comida y el algoritmo pierde diversidad. DE, gracias a su "
      "selección codiciosa individual y a la mutación con diferencias entre individuos, conserva diversidad durante más tiempo.")

# ---------------------------------------------------------------- Comparación global
R.h2("Comparación de modelos y validación cruzada repetida")
filas = []
for p, etq in [("svm_lineal", "SVM lineal"), ("svm_rbf", "SVM RBF"), ("rf", "RF")]:
    for a in ["Por defecto"] + ALGS:
        m = M[(p, a)]
        filas.append([etq, "Def." if a == "Por defecto" else a, pct(m.acc_cv), pct(m.exactitud),
                      f"{100*m.rcv_media:.2f} ± {100*m.rcv_std:.2f}"])
R.tabla("Exactitud por modelo con tres estimadores (%)",
        ["Modelo", "Alg.", "CV-5 (opt.)", "Prueba", "CV 10×5 (768)"], filas, [0.7, 0.45, 0.7, 0.55, 1.0],
        nota="CV-5 (opt.): valor de la función objetivo sobre el 80 %. CV 10×5: 50 particiones nuevas sobre todo el dataset.")

rcv = {(p, a): M[(p, a)].rcv_media for p in ["svm_lineal", "svm_rbf", "rf"] for a in ["Por defecto"] + ALGS}
mejor_rcv = max(rcv, key=rcv.get)
R.par("La Tabla VIII reúne tres estimaciones de la exactitud. Según la función objetivo, la SVM-RBF es el mejor modelo, "
      f"con {pct(best_rbf.acc_cv)} % frente a {pct(M[('rf','PSO')].acc_cv)} % del RF. En el conjunto de prueba el orden se "
      f"invierte: el RF de PSO/DE obtiene {pct(M[('rf','PSO')].exactitud)} % frente a {pct(M[('svm_rbf','PSO')].exactitud)} % "
      "de la SVM-RBF, y los modelos por defecto obtienen igual o mayor exactitud de prueba que los optimizados. Este "
      "resultado, aparentemente contradictorio, se debe a dos causas. Primero, el conjunto de prueba tiene solo 154 muestras: "
      "el intervalo de confianza al 95 % de una exactitud de 74 % es de aproximadamente ±7 puntos, mucho mayor que las "
      "diferencias observadas. Segundo, al optimizar sobre una sola partición de validación cruzada, los algoritmos "
      "seleccionan en parte el ruido de esa partición (sobreajuste a la validación), así que el valor de la función objetivo "
      "es una estimación optimista.")
R.par("La validación cruzada repetida 10×5 sobre las 768 muestras, con particiones que los algoritmos nunca vieron, es la "
      f"estimación más confiable. Con ella el mejor modelo es {'el RF' if mejor_rcv[0]=='rf' else 'la SVM ' + ('RBF' if mejor_rcv[0]=='svm_rbf' else 'lineal')} "
      f"({'por defecto' if mejor_rcv[1]=='Por defecto' else 'con ' + mejor_rcv[1]}), con {pct(rcv[mejor_rcv])} %. Todas las "
      f"configuraciones quedan entre {pct(min(rcv.values()))} % y {pct(max(rcv.values()))} %, un rango menor que una "
      "desviación estándar entre particiones. La optimización de hiperparámetros sí mejora la función objetivo que se le "
      "pidió minimizar, pero en este dataset la ganancia de generalización es pequeña, porque con los valores por defecto los modelos ya están cerca de la exactitud que permiten estas ocho variables.")

# ---------------------------------------------------------------- V. Conclusiones
R.h1("Conclusiones")
R.par("Se implementaron desde cero tres metaheurísticas, DA, PSO y DE, y se aplicaron a la búsqueda de hiperparámetros de una "
      "SVM (kernel lineal y RBF) y de un Random Forest para el Pima Indians Diabetes Dataset, minimizando el error de "
      "validación cruzada de cinco lotes. Las conclusiones principales son:")
R.viñeta(f"**Mejor modelo de IA: SVM con kernel RBF.** Obtuvo la mayor exactitud de validación ({pct(best_rbf.acc_cv)} %, "
         f"con *C* ≈ {pr('svm_rbf','DE')['C']:.1f} y *γ* ≈ {pr('svm_rbf','DE')['gamma']*1e4:.1f}×10^{{−4}}, encontrada por los tres algoritmos) y "
         f"también la mayor en la validación cruzada repetida 10×5 sobre datos no vistos ({pct(M[('svm_rbf','DE')].rcv_media)} % con "
         "los hiperparámetros de DE). En el conjunto de prueba de 154 muestras, el Random Forest optimizado con PSO o DE obtuvo "
         f"más exactitud ({pct(M[('rf','PSO')].exactitud)} %) y sensibilidad ({pct(M[('rf','PSO')].sensibilidad)} % frente a "
         f"{pct(M[('svm_rbf','DE')].sensibilidad)} %), así que es una alternativa razonable si se quiere priorizar la detección de "
         "casos positivos.")
R.viñeta("**Mejor algoritmo metaheurístico: DE.** En la SVM-RBF alcanzó el mejor valor conocido en las 10 corridas, con la menor "
         "dispersión, y en la corrida principal nunca fue superado. PSO fue competitivo, con la mejor media en el RF y "
         "convergencia rápida. DA, el algoritmo expuesto por el equipo, convergió pronto, pero tendió a estancarse en mesetas "
         "subóptimas por la pérdida de diversidad que provoca su esquema adaptativo de pesos.")
R.viñeta("**Generalización.** Las diferencias entre modelos y algoritmos son de uno a tres puntos porcentuales y quedan dentro "
         "del error de estimación de un dataset de 768 muestras. Optimizar sobre una sola partición de validación produce "
         "estimaciones optimistas; para un uso clínico convendría la validación cruzada anidada.")
R.viñeta("**Trabajo futuro.** Ampliar el intervalo de min_samples_leaf (el óptimo quedó en el límite), optimizar una métrica "
         "sensible al desbalance (F1 o exactitud balanceada) en lugar de la exactitud, y añadir a DA un mecanismo de "
         "reinicio o mutación para mantener la diversidad.")

# ---------------------------------------------------------------- Referencias
R.h5("Referencias")
for ref in [
    "S. Mirjalili, “Dragonfly algorithm: a new meta-heuristic optimization technique for solving single-objective, discrete, and multi-objective problems,” Neural Computing and Applications, vol. 27, no. 4, pp. 1053–1073, 2016.",
    "J. Kennedy and R. Eberhart, “Particle swarm optimization,” in Proc. IEEE Int. Conf. Neural Networks, vol. 4, pp. 1942–1948, 1995.",
    "Y. Shi and R. Eberhart, “A modified particle swarm optimizer,” in Proc. IEEE Int. Conf. Evolutionary Computation, pp. 69–73, 1998.",
    "R. Storn and K. Price, “Differential evolution – a simple and efficient heuristic for global optimization over continuous spaces,” Journal of Global Optimization, vol. 11, no. 4, pp. 341–359, 1997.",
    "F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.",
    "J. W. Smith, J. E. Everhart, W. C. Dickson, W. C. Knowler, and R. S. Johannes, “Using the ADAP learning algorithm to forecast the onset of diabetes mellitus,” in Proc. Annu. Symp. Computer Application in Medical Care, pp. 261–265, 1988.",
    "C. Cortes and V. Vapnik, “Support-vector networks,” Machine Learning, vol. 20, no. 3, pp. 273–297, 1995.",
    "L. Breiman, “Random forests,” Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.",
]:
    R.referencia(ref)

R.guardar(SALIDA)
print("OK", SALIDA)

# Optimización de hiperparámetros de SVM y Random Forest con los algoritmos Dragonfly, PSO y Evolución Diferencial para el Pima Indians Diabetes Dataset

| | | |
|:---:|:---:|:---:|
| **Marlene Ximena Sandoval García**<br>CUCEI, Universidad de Guadalajara<br>Guadalajara, México<br>Código: 219511456 | **Yael Iván García Mercado**<br>CUCEI, Universidad de Guadalajara<br>Guadalajara, México | **Marcos Daniel Díaz Briones**<br>CUCEI, Universidad de Guadalajara<br>Guadalajara, México |
| **Leonardo Daniel Palacios Palacios**<br>CUCEI, Universidad de Guadalajara<br>Guadalajara, México | **Oscar Javier Cervantes Cervera**<br>CUCEI, Universidad de Guadalajara<br>Guadalajara, México | |

---

***Resumen*—Este trabajo compara tres algoritmos metaheurísticos, el Dragonfly Algorithm (DA), la optimización por enjambre de partículas (PSO) y la evolución diferencial (DE), para ajustar los hiperparámetros de una máquina de soporte vectorial (SVM) con kernel lineal y RBF y de un Random Forest (RF) sobre el Pima Indians Diabetes Dataset. Los tres algoritmos se programaron desde cero y minimizan el error de validación cruzada estratificada de cinco lotes, fitness(x) = 1 − exactitud media. Con el mismo presupuesto de evaluaciones, los tres alcanzan la misma exactitud de validación en la SVM-RBF (79.31 %) y en la SVM lineal (78.82 %); en el RF, PSO y DE (78.51 %) superan a DA (78.35 %). En corridas repetidas con distintas semillas, DE alcanzó el mejor valor de la SVM-RBF en las 10 corridas y DA fue el que más se estancó. La SVM-RBF fue el mejor modelo tanto en la función objetivo como en una validación cruzada repetida 10×5 independiente, aunque en el conjunto de prueba el RF ajustado obtuvo mayor exactitud y sensibilidad. Las diferencias entre modelos son de uno a tres puntos porcentuales y quedan dentro de la variabilidad estadística que permite un conjunto de 768 muestras.**

***Palabras clave*—algoritmos metaheurísticos, Dragonfly Algorithm, PSO, evolución diferencial, optimización de hiperparámetros, SVM, random forest, diabetes**

---

## I. INTRODUCCIÓN

El desempeño de un modelo de aprendizaje automático depende en gran medida de sus hiperparámetros, es decir, de los valores que se fijan antes del entrenamiento y que el propio algoritmo de aprendizaje no ajusta. En una SVM, la constante de regularización *C* y el ancho del kernel *γ* controlan el compromiso entre margen y error de entrenamiento; en un Random Forest, el número de árboles y las reglas de crecimiento de cada árbol controlan la varianza del ensamble. Elegir mal estos valores produce modelos subajustados o sobreajustados.

La búsqueda en malla y la búsqueda aleatoria son los métodos más comunes, pero su costo crece rápidamente con la dimensión y la resolución del espacio de búsqueda. La función que relaciona hiperparámetros con error de validación no es diferenciable, es ruidosa y presenta mesetas, por lo que no admite métodos basados en gradiente. Los algoritmos metaheurísticos poblacionales, inspirados en procesos evolutivos o en el comportamiento colectivo de animales, son una alternativa natural para este tipo de problemas de caja negra [1]–[4].

El objetivo de este proyecto es diseñar, implementar y evaluar tres metaheurísticas para encontrar la configuración de hiperparámetros que minimiza el error de generalización estimado por validación cruzada. Se empleó el Dragonfly Algorithm (DA) [1], algoritmo que el equipo presentó en la exposición, junto con PSO [2], [3] y la evolución diferencial DE/rand/1/bin [4]. Los tres se programaron desde cero en Python con NumPy; scikit-learn [5] se usa solo para los modelos de clasificación y la validación cruzada. Se resuelven dos casos sobre el Pima Indians Diabetes Dataset [6]: una SVM [7] con kernel lineal y RBF y un Random Forest [8] con cuatro hiperparámetros enteros.

## II. CONJUNTO DE DATOS Y PREPROCESAMIENTO

El Pima Indians Diabetes Dataset [6] contiene 768 registros de mujeres de al menos 21 años de herencia pima, con ocho variables clínicas (embarazos, glucosa, presión arterial, grosor del pliegue cutáneo, insulina, IMC, función de pedigrí de diabetes y edad) y una etiqueta binaria que indica diagnóstico de diabetes (500 negativos y 268 positivos, 34.9 % de casos positivos).

En las columnas de glucosa, presión arterial, pliegue cutáneo, insulina e IMC un valor de cero es fisiológicamente imposible y representa un dato faltante (5, 35, 227, 374 y 11 registros, respectivamente). Estos ceros se reemplazaron por valores faltantes y se imputaron con la mediana. Para la SVM, además, las variables se estandarizaron a media cero y varianza unitaria. Ambas transformaciones forman parte de un *Pipeline* de scikit-learn, de modo que en cada lote de la validación cruzada se ajustan únicamente con los datos de entrenamiento de ese lote y no hay fuga de información.

Como pide el enunciado, para la SVM las etiquetas se transformaron del formato (0, 1) al formato bipolar (−1, +1), donde +1 corresponde a la clase diabética. Los datos se dividieron de forma estratificada en 80 % de entrenamiento (614 registros) y 20 % de prueba (154 registros: 100 negativos y 54 positivos). La optimización solo usa el conjunto de entrenamiento; el de prueba se reserva para medir las métricas finales.

## III. METODOLOGÍA

### A. Función objetivo

Cada vector de decisión **x** se evalúa con validación cruzada estratificada de *K* = 5 lotes sobre el conjunto de entrenamiento, y se minimiza

$$
\text{fitness}(\mathbf{x}) = 1 - \frac{1}{K}\sum_{k=1}^{K} \text{Accuracy}_k(\mathbf{x}) \tag{1}
$$

Las particiones de la validación cruzada se fijaron con una semilla, de modo que la función objetivo es determinista y los tres algoritmos optimizan exactamente la misma función. Como cada lote tiene 122 o 123 muestras, la exactitud solo cambia en saltos de aproximadamente 0.16 %, por lo que la superficie de búsqueda está formada por mesetas. Para no reentrenar modelos idénticos, la función guarda en caché el resultado de cada combinación de hiperparámetros; las evaluaciones en caché se cuentan igualmente para el presupuesto.

### B. Vectores de decisión y espacios de búsqueda

El vector de decisión depende del modelo. En la SVM con kernel lineal, *K*(**u**, **v**) = **u**ᵀ**v** no depende de *γ*, de modo que el vector se reduce a **x** = [*C*]. Con kernel RBF, *K*(**u**, **v**) = exp(−*γ*‖**u** − **v**‖²), y el vector es **x** = [*C*, *γ*]. Los intervalos son los del enunciado, *C* ∈ [0.1, 100] y *γ* ∈ [10⁻⁴, 10]. Como estos intervalos abarcan tres y cinco órdenes de magnitud, los algoritmos los recorren en escala logarítmica, es decir, buscan log₁₀*C* ∈ [−1, 2] y log₁₀*γ* ∈ [−4, 1] y decodifican *C* = 10^*x*₁ y *γ* = 10^*x*₂. El espacio es el mismo, pero así el intervalo [0.1, 1] recibe la misma atención que [10, 100].

En el Random Forest, **x** = [n_estimators, max_depth, min_samples_split, min_samples_leaf], con los cuatro valores enteros en [10, 300], [3, 30], [2, 20] y [1, 20]. Los algoritmos trabajan en el continuo [*l* − 0.5, *u* + 0.5] y cada componente se redondea al entero más cercano antes de evaluar, lo que da a los extremos la misma probabilidad que a los valores interiores. La Tabla I resume ambos espacios.

**TABLA I. VECTORES DE DECISIÓN Y ESPACIOS DE BÚSQUEDA**

| Modelo | Vector **x** | Intervalos | Tipo / codificación |
|:---|:---|:---|:---|
| SVM lineal | [*C*] | *C* ∈ [0.1, 100] | Real, log₁₀ |
| SVM RBF | [*C*, *γ*] | *C* ∈ [0.1, 100], *γ* ∈ [10⁻⁴, 10] | Real, log₁₀ |
| Random Forest | [*n*_est, *d*_max, *s*_min, *l*_min] | [10, 300], [3, 30], [2, 20], [1, 20] | Entero (redondeo) |

### C. Dragonfly Algorithm (DA)

El DA [1] imita el enjambre estático (caza) y dinámico (migración) de las libélulas. Cada libélula *i* tiene una posición **X**ᵢ y un vector de paso Δ**X**ᵢ. Respecto a sus vecinos *j* = 1,…,*N* dentro de un radio *r* se calculan cinco comportamientos: separación **S**ᵢ = −Σ(**X**ᵢ − **X**ⱼ), alineación **A**ᵢ = media(Δ**X**ⱼ), cohesión **C**ᵢ = media(**X**ⱼ) − **X**ᵢ, atracción a la comida **F**ᵢ = **X**⁺ − **X**ᵢ y distracción del enemigo **E**ᵢ = **X**⁻ + **X**ᵢ, donde **X**⁺ y **X**⁻ son la mejor y la peor solución encontradas. El paso y la posición se actualizan con

$$
\Delta\mathbf{X}_{t+1} = s\mathbf{S} + a\mathbf{A} + c\mathbf{C} + f\mathbf{F} + e\mathbf{E} + w\,\Delta\mathbf{X}_t \tag{2}
$$

$$
\mathbf{X}_{t+1} = \mathbf{X}_t + \Delta\mathbf{X}_{t+1} \tag{3}
$$

Si una libélula no tiene vecinos, realiza un vuelo de Lévy, **X**ₜ₊₁ = **X**ₜ + Lévy(*d*)·**X**ₜ, generado con el algoritmo de Mantegna (*β* = 1.5). En nuestra implementación el paso de Lévy se escala por el ancho del intervalo de cada variable para que sea comparable entre dimensiones con rangos muy distintos. Los pesos son adaptativos, como en el artículo original: *w* decrece linealmente de 0.9 a 0.4; *s*, *a* y *c* se toman como 2·rand·*c*ₜ, *f* = 2·rand y *e* = *c*ₜ, con *c*ₜ = 0.1 − 0.2*t*/*T* (acotado en 0), y el radio de vecindad crece de un cuarto del rango hasta cubrir todo el espacio. Así, al inicio domina la exploración y al final la explotación. El paso se acota a ±10 % del rango.

### D. Optimización por enjambre de partículas (PSO)

En PSO [2], [3] cada partícula recuerda su mejor posición **p**ᵢ y conoce la mejor del enjambre **g**. Se usó la variante global con peso de inercia linealmente decreciente:

$$
\mathbf{v}_{t+1} = w\mathbf{v}_t + c_1 r_1(\mathbf{p}_i - \mathbf{x}_t) + c_2 r_2(\mathbf{g} - \mathbf{x}_t) \tag{4}
$$

$$
\mathbf{x}_{t+1} = \mathbf{x}_t + \mathbf{v}_{t+1} \tag{5}
$$

con *w* de 0.9 a 0.4, *c*₁ = *c*₂ = 2, *r*₁, *r*₂ ~ U(0, 1) y la velocidad limitada a ±20 % del rango. Las posiciones que salen del espacio se proyectan al límite más cercano.

### E. Evolución diferencial (DE/rand/1/bin)

DE [4] es un algoritmo evolutivo que genera un vector mutante a partir de tres individuos distintos elegidos al azar, **v** = **x**ᵣ₁ + *F*(**x**ᵣ₂ − **x**ᵣ₃). Después, la cruza binomial forma el vector de prueba **u** tomando cada componente de **v** con probabilidad *CR* (al menos una componente siempre viene de **v**). Por último, la selección codiciosa reemplaza a **x**ᵢ por **u** si *f*(**u**) ≤ *f*(**x**ᵢ). Se usaron *F* = 0.5 y *CR* = 0.9.

### F. Configuración experimental

Para que la comparación sea justa, los tres algoritmos usan la misma población inicial (misma semilla), el mismo tamaño de población y el mismo número de iteraciones, es decir, el mismo número de evaluaciones de la función objetivo (Tabla II). La corrida principal usa la semilla 42. Para medir la robustez se repitió la optimización con semillas adicionales (10 corridas en total para la SVM-RBF, 5 para la SVM lineal y 3 para el RF).

**TABLA II. PARÁMETROS DE LOS ALGORITMOS**

| Parámetro | SVM (lineal y RBF) | Random Forest |
|:---|:---|:---|
| Población / iteraciones | 20 / 40 | 15 / 30 |
| Evaluaciones por corrida | 820 | 465 |
| DA | *w*: 0.9→0.4, *c*ₜ: 0.1→0, Lévy *β* = 1.5 | igual |
| PSO | *w*: 0.9→0.4, *c*₁ = *c*₂ = 2, *v*max = 0.2·rango | igual |
| DE | DE/rand/1/bin, *F* = 0.5, *CR* = 0.9 | igual |

Con los mejores hiperparámetros de cada algoritmo se entrenó el modelo sobre las 614 muestras de entrenamiento y se evaluó sobre las 154 de prueba. Tomando la clase diabética como positiva, se reportan exactitud (TP+TN)/N, precisión TP/(TP+FP), sensibilidad TP/(TP+FN), especificidad TN/(TN+FP), F1 y la matriz de confusión. Como referencia se incluye cada modelo con los hiperparámetros por defecto de scikit-learn (SVM: *C* = 1, *γ* = 'scale'; RF: 100 árboles sin límite de profundidad). Además, como el conjunto de prueba es pequeño (una muestra equivale a 0.65 %), cada configuración final se evaluó con validación cruzada estratificada repetida 10×5 sobre las 768 muestras, con particiones distintas a las usadas en la optimización.

## IV. RESULTADOS

### A. SVM con kernel lineal y RBF

Las Figs. 1 y 2 muestran las curvas de convergencia (mejor fitness acumulado por iteración) de la corrida principal. En el kernel lineal el problema es unidimensional y la exactitud de validación es prácticamente constante para *C* ≳ 1, por lo que los tres algoritmos ya tenían el óptimo de 78.82 % en la población inicial (iteración 0) y la curva es plana. En el kernel RBF los tres algoritmos parten de la misma población y convergen al mismo valor, fitness = 0.2069 (exactitud de validación de 79.31 %). DE lo alcanzó en la iteración 8, PSO en la 10 y DA en la 11.

> *[Figura omitida en el .md: curvas de convergencia de DA, PSO y DE, fitness vs. iteración (0–40). En el kernel lineal las tres curvas son planas en 0.2118.]*
>
> **Fig. 1.** Curvas de convergencia de DA, PSO y DE para la SVM con kernel lineal (semilla 42).

> *[Figura omitida en el .md: curvas de convergencia de DA, PSO y DE, fitness vs. iteración (0–40). Las tres convergen a 0.2069.]*
>
> **Fig. 2.** Curvas de convergencia de DA, PSO y DE para la SVM con kernel RBF (semilla 42).

**TABLA III. HIPERPARÁMETROS ÓPTIMOS DE LA SVM (SEMILLA 42)**

| Kernel | Alg. | *C* | *γ* | Acc. CV (%) |
|:---|:---|---:|---:|---:|
| Lineal | DA | 20.983 | — | 78.82 |
| Lineal | PSO | 20.983 | — | 78.82 |
| Lineal | DE | 32.590 | — | 78.82 |
| RBF | DA | 49.743 | 1.18×10⁻³ | 79.31 |
| RBF | PSO | 100.000 | 8.08×10⁻⁴ | 79.31 |
| RBF | DE | 85.493 | 9.34×10⁻⁴ | 79.31 |

La Tabla III muestra que en el kernel RBF los tres algoritmos encontraron puntos distintos de una misma región: *C* alto (50–100) con *γ* ≈ 10⁻³. Con *γ* tan pequeño, el kernel RBF se comporta casi como un kernel lineal muy suave, lo que concuerda con que el kernel lineal también alcance casi la misma exactitud. Las corridas con otras semillas (Sección IV-C) revelaron una segunda región con el mismo fitness, *C* ≈ 0.8 y *γ* ≈ 0.06, que corresponde a un modelo más regularizado y más no lineal. Es decir, la función objetivo tiene varios óptimos equivalentes.

**TABLA IV. MÉTRICAS DE LA SVM EN EL CONJUNTO DE PRUEBA (%)**

| Kernel | Alg. | Exact. | Prec. | Sens. | Espec. | F1 |
|:---|:---|---:|---:|---:|---:|---:|
| Lineal | Def. | 70.13 | 59.09 | 48.15 | 82.00 | 53.06 |
| Lineal | DA | 70.13 | 59.09 | 48.15 | 82.00 | 53.06 |
| Lineal | PSO | 70.13 | 59.09 | 48.15 | 82.00 | 53.06 |
| Lineal | DE | 70.13 | 59.09 | 48.15 | 82.00 | 53.06 |
| RBF | Def. | 74.03 | 65.22 | 55.56 | 84.00 | 60.00 |
| RBF | DA | 70.13 | 59.09 | 48.15 | 82.00 | 53.06 |
| RBF | PSO | 70.78 | 60.47 | 48.15 | 83.00 | 53.61 |
| RBF | DE | 70.78 | 60.47 | 48.15 | 83.00 | 53.61 |

*a. Def.: hiperparámetros por defecto de scikit-learn. Clase positiva: diabética (+1).*

Las métricas de prueba (Tabla IV y Figs. 3 y 4) son casi idénticas entre algoritmos. En el kernel lineal los tres producen las mismas predicciones que el modelo con *C* = 1: exactitud de 70.13 %, precisión de 59.09 % y sensibilidad de 48.15 %. En el kernel RBF, PSO y DE obtienen 70.78 % de exactitud (un falso positivo menos que DA). La sensibilidad, de 48.15 %, es baja en todos los casos: el modelo detecta menos de la mitad de las pacientes diabéticas, un efecto del desbalance de clases y de que la función objetivo optimiza la exactitud y no la sensibilidad.

**Fig. 3.** Matrices de confusión en prueba de la SVM lineal con los hiperparámetros de DA, PSO y DE (filas: real; columnas: predicho).

| Real \ Predicho | DA: No diab. | DA: Diab. | PSO: No diab. | PSO: Diab. | DE: No diab. | DE: Diab. |
|:---|---:|---:|---:|---:|---:|---:|
| **No diab.** | 82 | 18 | 82 | 18 | 82 | 18 |
| **Diab.** | 28 | 26 | 28 | 26 | 28 | 26 |

**Fig. 4.** Matrices de confusión en prueba de la SVM RBF con los hiperparámetros de DA, PSO y DE (filas: real; columnas: predicho).

| Real \ Predicho | DA: No diab. | DA: Diab. | PSO: No diab. | PSO: Diab. | DE: No diab. | DE: Diab. |
|:---|---:|---:|---:|---:|---:|---:|
| **No diab.** | 82 | 18 | 83 | 17 | 83 | 17 |
| **Diab.** | 28 | 26 | 28 | 26 | 28 | 26 |

### B. Random Forest

La Fig. 5 muestra la convergencia para el RF. PSO encontró su mejor configuración en la iteración 14 y DE en la 6, ambas con fitness = 0.2149 (78.51 % de exactitud de validación). DA mejoró en la iteración 6 y después se estancó en 0.2165 (78.35 %). En este problema DA necesitó menos tiempo de cómputo (191 s frente a 390 s de PSO y 344 s de DE) porque su población se concentró pronto en pocas configuraciones: solo 284 de las 465 evaluaciones fueron combinaciones nuevas. Es otra señal de que perdió diversidad.

> *[Figura omitida en el .md: curvas de convergencia de DA, PSO y DE, fitness vs. iteración (0–30). DE y PSO llegan a 0.2149; DA se estanca en 0.2165.]*
>
> **Fig. 5.** Curvas de convergencia de DA, PSO y DE para el Random Forest (semilla 42).

**TABLA V. HIPERPARÁMETROS ÓPTIMOS DEL RANDOM FOREST (SEMILLA 42)**

| Alg. | n_estim. | max_depth | min_split | min_leaf | Acc. CV (%) |
|:---|---:|---:|---:|---:|---:|
| DA | 39 | 17 | 7 | 20 | 78.35 |
| PSO | 165 | 22 | 16 | 20 | 78.51 |
| DE | 165 | 18 | 20 | 20 | 78.51 |

En la Tabla V, los tres algoritmos llevaron min_samples_leaf al límite superior del intervalo (20 muestras por hoja). Es decir, en este dataset conviene regularizar fuertemente cada árbol, y es posible que el óptimo esté fuera del espacio de búsqueda definido. PSO y DE coincidieron en 165 árboles; la profundidad máxima y min_samples_split resultaron poco relevantes, porque con hojas de 20 muestras los árboles rara vez alcanzan la profundidad máxima.

**TABLA VI. MÉTRICAS DEL RANDOM FOREST EN EL CONJUNTO DE PRUEBA (%)**

| Alg. | Exact. | Prec. | Sens. | Espec. | F1 |
|:---|---:|---:|---:|---:|---:|
| Def. | 76.62 | 69.57 | 59.26 | 86.00 | 64.00 |
| DA | 72.73 | 65.00 | 48.15 | 86.00 | 55.32 |
| PSO | 74.68 | 68.29 | 51.85 | 87.00 | 58.95 |
| DE | 74.68 | 68.29 | 51.85 | 87.00 | 58.95 |

Con los hiperparámetros de PSO y DE, el RF obtuvo 74.68 % de exactitud, 68.29 % de precisión, 51.85 % de sensibilidad y 87.00 % de especificidad (Tabla VI y Fig. 6), mejor que DA en todas las métricas. Estos son los mejores valores de prueba entre los modelos optimizados.

**Fig. 6.** Matrices de confusión en prueba del Random Forest con los hiperparámetros de DA, PSO y DE (filas: real; columnas: predicho).

| Real \ Predicho | DA: No diab. | DA: Diab. | PSO: No diab. | PSO: Diab. | DE: No diab. | DE: Diab. |
|:---|---:|---:|---:|---:|---:|---:|
| **No diab.** | 86 | 14 | 87 | 13 | 87 | 13 |
| **Diab.** | 28 | 26 | 26 | 28 | 26 | 28 |

### C. Robustez frente a la semilla aleatoria

Una sola corrida no basta para comparar algoritmos estocásticos. La Tabla VII resume las corridas con distintas semillas: la media y la desviación estándar de la mejor exactitud de validación, y la iteración media en la que se encontró el mejor valor.

**TABLA VII. RESULTADOS DE CORRIDAS INDEPENDIENTES (EXACTITUD CV, %)**

| Modelo | Alg. | *n* | Media ± d.e. | Máx. | Iter. |
|:---|:---|---:|:---:|---:|---:|
| SVM lin. | DA | 5 | 78.82 ± 0.00 | 78.82 | 0.0 |
| SVM lin. | PSO | 5 | 78.82 ± 0.00 | 78.82 | 0.0 |
| SVM lin. | DE | 5 | 78.82 ± 0.00 | 78.82 | 0.0 |
| SVM RBF | DA | 10 | 79.05 ± 0.19 | 79.31 | 10.7 |
| SVM RBF | PSO | 10 | 79.26 ± 0.11 | 79.31 | 7.4 |
| SVM RBF | DE | 10 | 79.31 ± 0.00 | 79.31 | 8.2 |
| RF | DA | 3 | 78.34 ± 0.16 | 78.51 | 11.3 |
| RF | PSO | 3 | 78.67 ± 0.28 | 78.99 | 10.7 |
| RF | DE | 3 | 78.62 ± 0.19 | 78.83 | 12.7 |

En la SVM-RBF, el mejor valor conocido (79.31 %) lo alcanzó DE en 10 de 10 corridas, PSO en 8 de 10 y DA en 3 de 10. DA quedó atrapado con frecuencia en una meseta de 78.99 % con *C* ≈ 5 y *γ* ≈ 5×10⁻³. En el RF, el mejor valor (78.99 %) lo alcanzó DE en 0 de 3 corridas, PSO en 1 de 3 y DA en 0 de 3; ese valor corresponde a un bosque muy distinto al de la corrida principal (10 árboles de profundidad 4), otra muestra de que hay varios óptimos casi equivalentes. Por medias, PSO (78.67 %) y DE (78.62 %) superan a DA (78.34 %). El comportamiento de DA se explica por su diseño: los pesos de separación, alineación y cohesión se anulan en la segunda mitad de la ejecución y el radio de vecindad crece hasta abarcar a todo el enjambre, de modo que en problemas con mesetas, donde la mejor solución deja de cambiar, todas las libélulas se agrupan alrededor de la comida y el algoritmo pierde diversidad. DE, gracias a su selección codiciosa individual y a la mutación con diferencias entre individuos, conserva diversidad durante más tiempo.

### D. Comparación de modelos y validación cruzada repetida

**TABLA VIII. EXACTITUD POR MODELO CON TRES ESTIMADORES (%)**

| Modelo | Alg. | CV-5 (opt.) | Prueba | CV 10×5 (768) |
|:---|:---|---:|---:|:---:|
| SVM lineal | Def. | 78.82 | 70.13 | 76.46 ± 3.03 |
| SVM lineal | DA | 78.82 | 70.13 | 76.43 ± 3.03 |
| SVM lineal | PSO | 78.82 | 70.13 | 76.43 ± 3.03 |
| SVM lineal | DE | 78.82 | 70.13 | 76.43 ± 3.03 |
| SVM RBF | Def. | 78.01 | 74.03 | 76.03 ± 3.01 |
| SVM RBF | DA | 79.31 | 70.13 | 76.56 ± 3.25 |
| SVM RBF | PSO | 79.31 | 70.78 | 76.68 ± 3.37 |
| SVM RBF | DE | 79.31 | 70.78 | 76.71 ± 3.35 |
| RF | Def. | 76.06 | 76.62 | 76.03 ± 3.78 |
| RF | DA | 78.35 | 72.73 | 75.89 ± 3.21 |
| RF | PSO | 78.51 | 74.68 | 75.68 ± 3.08 |
| RF | DE | 78.51 | 74.68 | 75.68 ± 3.08 |

*b. CV-5 (opt.): valor de la función objetivo sobre el 80 %. CV 10×5: 50 particiones nuevas sobre todo el dataset.*

La Tabla VIII reúne tres estimaciones de la exactitud. Según la función objetivo, la SVM-RBF es el mejor modelo, con 79.31 % frente a 78.51 % del RF. En el conjunto de prueba el orden se invierte: el RF de PSO/DE obtiene 74.68 % frente a 70.78 % de la SVM-RBF, y los modelos por defecto obtienen igual o mayor exactitud de prueba que los optimizados. Este resultado, aparentemente contradictorio, se debe a dos causas. Primero, el conjunto de prueba tiene solo 154 muestras: el intervalo de confianza al 95 % de una exactitud de 74 % es de aproximadamente ±7 puntos, mucho mayor que las diferencias observadas. Segundo, al optimizar sobre una sola partición de validación cruzada, los algoritmos seleccionan en parte el ruido de esa partición (sobreajuste a la validación), así que el valor de la función objetivo es una estimación optimista.

La validación cruzada repetida 10×5 sobre las 768 muestras, con particiones que los algoritmos nunca vieron, es la estimación más confiable. Con ella el mejor modelo es la SVM RBF (con DE), con 76.71 %. Todas las configuraciones quedan entre 75.68 % y 76.71 %, un rango menor que una desviación estándar entre particiones. La optimización de hiperparámetros sí mejora la función objetivo que se le pidió minimizar, pero en este dataset la ganancia de generalización es pequeña, porque con los valores por defecto los modelos ya están cerca de la exactitud que permiten estas ocho variables.

## V. CONCLUSIONES

Se implementaron desde cero tres metaheurísticas, DA, PSO y DE, y se aplicaron a la búsqueda de hiperparámetros de una SVM (kernel lineal y RBF) y de un Random Forest para el Pima Indians Diabetes Dataset, minimizando el error de validación cruzada de cinco lotes. Las conclusiones principales son:

- **Mejor modelo de IA: SVM con kernel RBF.** Obtuvo la mayor exactitud de validación (79.31 %, con *C* ≈ 85.5 y *γ* ≈ 9.3×10⁻⁴, encontrada por los tres algoritmos) y también la mayor en la validación cruzada repetida 10×5 sobre datos no vistos (76.71 % con los hiperparámetros de DE). En el conjunto de prueba de 154 muestras, el Random Forest optimizado con PSO o DE obtuvo más exactitud (74.68 %) y sensibilidad (51.85 % frente a 48.15 %), así que es una alternativa razonable si se quiere priorizar la detección de casos positivos.
- **Mejor algoritmo metaheurístico: DE.** En la SVM-RBF alcanzó el mejor valor conocido en las 10 corridas, con la menor dispersión, y en la corrida principal nunca fue superado. PSO fue competitivo, con la mejor media en el RF y convergencia rápida. DA, el algoritmo expuesto por el equipo, convergió pronto, pero tendió a estancarse en mesetas subóptimas por la pérdida de diversidad que provoca su esquema adaptativo de pesos.
- **Generalización.** Las diferencias entre modelos y algoritmos son de uno a tres puntos porcentuales y quedan dentro del error de estimación de un dataset de 768 muestras. Optimizar sobre una sola partición de validación produce estimaciones optimistas; para un uso clínico convendría la validación cruzada anidada.
- **Trabajo futuro.** Ampliar el intervalo de min_samples_leaf (el óptimo quedó en el límite), optimizar una métrica sensible al desbalance (F1 o exactitud balanceada) en lugar de la exactitud, y añadir a DA un mecanismo de reinicio o mutación para mantener la diversidad.

## REFERENCIAS

[1] S. Mirjalili, “Dragonfly algorithm: a new meta-heuristic optimization technique for solving single-objective, discrete, and multi-objective problems,” *Neural Computing and Applications*, vol. 27, no. 4, pp. 1053–1073, 2016.

[2] J. Kennedy and R. Eberhart, “Particle swarm optimization,” in *Proc. IEEE Int. Conf. Neural Networks*, vol. 4, pp. 1942–1948, 1995.

[3] Y. Shi and R. Eberhart, “A modified particle swarm optimizer,” in *Proc. IEEE Int. Conf. Evolutionary Computation*, pp. 69–73, 1998.

[4] R. Storn and K. Price, “Differential evolution – a simple and efficient heuristic for global optimization over continuous spaces,” *Journal of Global Optimization*, vol. 11, no. 4, pp. 341–359, 1997.

[5] F. Pedregosa *et al.*, “Scikit-learn: Machine learning in Python,” *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[6] J. W. Smith, J. E. Everhart, W. C. Dickson, W. C. Knowler, and R. S. Johannes, “Using the ADAP learning algorithm to forecast the onset of diabetes mellitus,” in *Proc. Annu. Symp. Computer Application in Medical Care*, pp. 261–265, 1988.

[7] C. Cortes and V. Vapnik, “Support-vector networks,” *Machine Learning*, vol. 20, no. 3, pp. 273–297, 1995.

[8] L. Breiman, “Random forests,” *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.

---

*Algoritmos Metaheurísticos · Sección D02 · Prof. Mario Antonio Ruz Canul · CUCEI, UdG · 2026*

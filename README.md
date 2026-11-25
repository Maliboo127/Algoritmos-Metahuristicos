# 🐉 Proyecto Final: Algoritmos Metaheurísticos

<h2 align="center">Obtención de hiperparámetros óptimos con DA, PSO y DE para SVM y Random Forest</h2>

<p align="center">
  <img src="https://img.shields.io/badge/Proyecto-Algoritmos%20Metaheurísticos-0A2540?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.13.1-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-Optimización-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Machine%20Learning-SVM%20%7C%20RF-2563EB?style=for-the-badge" />
</p>

---

## 👥 Equipo DragonFly

<p align="center">
  <strong>Universidad de Guadalajara · CUCEI</strong><br>
  <strong>Algoritmos Metaheurísticos Sección D02</strong><br>
  <strong>Prof. Mario Antonio Ruz Canul </strong><br>
</p>

| Integrante | |
|:---|:---:|
| 👩‍💻 **Marlene Ximena Sandoval García** | 
| 👨‍💻 **Yael Iván García Mercado** | 
| 👨‍💻 **Marcos Daniel Díaz Briones** | 
| 👨‍💻 **Leonardo Daniel Palacios Palacios** | 
| 👨‍💻 **Oscar Javier Cervantes Cervera** | 

---

## 📁 Estructura y archivos del proyecto

| Archivo | Descripción |
|:---|:---|
| `metaheuristicas.py` | **Dragonfly Algorithm**, **PSO** y **Evolución Diferencial (DE/rand/1/bin)**, programados desde cero con NumPy. |
| `datos_y_objetivos.py` | Carga del dataset, imputación de ceros, etiquetas bipolares (−1, +1) para la SVM, espacios de búsqueda y función `fitness(x) = 1 − media(Accuracy_k)` con CV estratificada de k = 5. |
| `ejecutar_optimizacion.py` | Corre los 3 algoritmos sobre `svm_lineal`, `svm_rbf` o `rf` y guarda las curvas y los mejores hiperparámetros en `resultados/*.json`. |
| `evaluar_y_graficar.py` | Entrena con los hiperparámetros óptimos, calcula exactitud, precisión, sensibilidad, especificidad, F1 y matriz de confusión en prueba, la CV repetida 10×5, las gráficas de convergencia y las matrices. |
| `diabetes_dataset.csv` | Pima Indians Diabetes Dataset (768 × 9). |
| `resultados/` | JSON de cada corrida, `metricas.csv`, `robustez.csv` y figuras PNG. |
| `logs/` | Salida de consola de las corridas. |
| `reporte/generar_reporte.py`, `reporte/docxkit.py` | Generan el reporte en la plantilla IEEE a partir de los resultados. |

> 💡 **Nota:** scikit-learn se usa **solo** para los modelos (SVC, RandomForestClassifier), el preprocesamiento y la validación cruzada.  
> Los algoritmos de optimización no usan ninguna librería de optimización.

---

## 🚀 Cómo reproducir el proyecto

### 1. Instalación de dependencias

```bash
pip install numpy pandas scikit-learn matplotlib tabulate
```

### 2. Ejecución de los algoritmos

**Corrida principal (semilla 42):**

```bash
python ejecutar_optimizacion.py svm_lineal
python ejecutar_optimizacion.py svm_rbf
python ejecutar_optimizacion.py rf
```

**Corridas extra para la tabla de robustez (opcional):**

```bash
python ejecutar_optimizacion.py svm_rbf    --semillas 1 2 3 4 5 6 7 8 9
python ejecutar_optimizacion.py svm_lineal --semillas 1 2 3 4
python ejecutar_optimizacion.py rf         --semillas 1 2
```

### 3. Obtención de métricas y gráficas

```bash
python evaluar_y_graficar.py
```

### ⏱️ Tiempos aproximados de ejecución

<sub>En una CPU de 2 núcleos:</sub>

| Modelo | Tiempo aproximado por algoritmo |
|:---|:---:|
| SVM-RBF | ≈ 35 s |
| SVM lineal | ≈ 2–3 min |
| Random Forest | ≈ 3–6 min |

---

## ⚙️ Configuración experimental

### 📊 Partición de datos

* Partición estratificada 80/20 (`random_state=42`). La optimización solo ve el 80 %.

### 🔍 Espacios de búsqueda

**Máquinas de soporte vectorial (SVM)**

* SVM: lineal `x = [C]` y RBF `x = [C, γ]`, con `C ∈ [0.1, 100]` y `γ ∈ [1e-4, 10]`, buscados en escala log10.

**Random Forest (RF)**

* RF: `x = [n_estimators, max_depth, min_samples_split, min_samples_leaf]`, enteros en `[10,300] [3,30] [2,20] [1,20]` (redondeo).

### 🧮 Presupuesto de evaluación

* Presupuesto igual para los 3 algoritmos: SVM con 20 individuos × 40 iteraciones (820 evaluaciones); RF con 15 × 30 (465 evaluaciones).

### 🧬 Parámetros de los algoritmos metaheurísticos

| Algoritmo | Parámetros |
|:---|:---|
| 🐉 **DA (Dragonfly Algorithm)** | w de 0.9 a 0.4 y pesos s, a, c, e adaptativos, con vuelo de Lévy (β = 1.5). |
| 🌐 **PSO (Particle Swarm Optimization)** | w de 0.9 a 0.4, c1 = c2 = 2. |
| 🧬 **DE (Differential Evolution)** | F = 0.5, CR = 0.9. |

---

<p align="center">
  <strong>🐉 DragonFly · Proyecto Final · Algoritmos Metaheurísticos</strong><br>
  <sub>CUCEI · Universidad de Guadalajara · Sección D02</sub>
</p>

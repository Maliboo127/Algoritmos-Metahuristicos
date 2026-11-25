"""
metaheuristicas.py
------------------
Implementación desde cero (solo NumPy) de tres algoritmos metaheurísticos
para minimización con restricciones de caja:

    * DA  - Dragonfly Algorithm (Mirjalili, 2016)       -> algoritmo expuesto por el equipo
    * PSO - Particle Swarm Optimization (Kennedy & Eberhart, 1995; Shi & Eberhart, 1998)
    * DE  - Differential Evolution, DE/rand/1/bin (Storn & Price, 1997)

Todos comparten la misma interfaz:

    resultado = algoritmo(f, lb, ub, n_pop=..., max_iter=..., seed=...)

y devuelven un diccionario con:
    best_x  : mejor vector de decisión encontrado
    best_f  : mejor valor de la función objetivo
    curva   : mejor valor acumulado (best-so-far) en cada iteración (len = max_iter + 1)
    n_eval  : número de evaluaciones de la función objetivo

La iteración 0 de la curva corresponde a la población inicial, de modo que los
tres algoritmos usan exactamente el mismo presupuesto: n_pop * (max_iter + 1)
evaluaciones.
"""

from __future__ import annotations

import math
import numpy as np


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _evaluar(f, X):
    return np.array([f(x) for x in X], dtype=float)


def _levy(d, rng, beta=1.5):
    """Paso de vuelo de Lévy (algoritmo de Mantegna), igual que en el DA original."""
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    r1 = rng.random(d)
    r2 = rng.random(d)
    return 0.01 * r1 * sigma / np.abs(r2) ** (1 / beta)


# ---------------------------------------------------------------------------
# 1) Dragonfly Algorithm (DA)
# ---------------------------------------------------------------------------
def dragonfly(f, lb, ub, n_pop=20, max_iter=50, seed=None, verbose=False):
    """
    Dragonfly Algorithm para problemas continuos (Mirjalili, 2016).

    Cada libélula i tiene posición X_i y paso ΔX_i. En cada iteración se calculan
    los cinco comportamientos de enjambre respecto a sus vecinos (radio r):
        S  separación   = -Σ (X - X_j)
        A  alineación   = media(ΔX_j)
        C  cohesión     = media(X_j) - X
        F  comida       = X⁺ - X     (mejor solución encontrada)
        E  enemigo      = X⁻ + X     (peor solución encontrada)
    y se actualiza
        ΔX(t+1) = s·S + a·A + c·C + f·F + e·E + w·ΔX(t)
        X(t+1)  = X(t) + ΔX(t+1)
    Si una libélula no tiene vecinos, se mueve con un vuelo de Lévy:
        X(t+1)  = X(t) + Lévy(d)·X(t)

    Los pesos se ajustan de forma adaptativa: w decrece de 0.9 a 0.4, los pesos
    s, a, c y e decrecen hasta 0 (más exploración al inicio) y el radio de
    vecindad crece con las iteraciones (más explotación al final).
    """
    rng = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    d = lb.size
    rango = ub - lb
    delta_max = rango / 10.0

    X = lb + rng.random((n_pop, d)) * rango
    dX = lb + rng.random((n_pop, d)) * rango          # pasos iniciales (como en el código original)
    dX = np.clip(dX - (lb + ub) / 2, -delta_max, delta_max)

    fit = _evaluar(f, X)
    n_eval = n_pop
    i_best, i_worst = int(np.argmin(fit)), int(np.argmax(fit))
    food_x, food_f = X[i_best].copy(), fit[i_best]
    enemy_x, enemy_f = X[i_worst].copy(), fit[i_worst]
    curva = [food_f]

    for t in range(1, max_iter + 1):
        # parámetros adaptativos
        r = rango / 4 + rango * (t / max_iter) * 2
        w = 0.9 - t * (0.9 - 0.4) / max_iter
        my_c = max(0.1 - t * (0.1 / (max_iter / 2)), 0.0)
        s = 2 * rng.random() * my_c
        a = 2 * rng.random() * my_c
        c = 2 * rng.random() * my_c
        fw = 2 * rng.random()
        e = my_c

        X_new = X.copy()
        dX_new = dX.copy()
        for i in range(n_pop):
            dist = np.abs(X - X[i])
            vecinos = np.all(dist <= r, axis=1) & (np.arange(n_pop) != i)
            nv = int(vecinos.sum())

            if nv > 0:
                Xn, dXn = X[vecinos], dX[vecinos]
                S = -np.sum(X[i] - Xn, axis=0)
                A = np.mean(dXn, axis=0)
                C = np.mean(Xn, axis=0) - X[i]
            else:
                S = np.zeros(d); A = dX[i].copy(); C = np.zeros(d)

            F = food_x - X[i] if np.all(np.abs(X[i] - food_x) <= r) else np.zeros(d)
            E = enemy_x + X[i] if np.all(np.abs(X[i] - enemy_x) <= r) else np.zeros(d)

            if nv > 0:
                paso = s * S + a * A + c * C + fw * F + e * E + w * dX[i]
                paso = np.clip(paso, -delta_max, delta_max)
                X_new[i] = X[i] + paso
                dX_new[i] = paso
            else:
                # vuelo de Lévy (escalado por el rango para que el paso sea relevante en todas las dimensiones)
                X_new[i] = X[i] + _levy(d, rng) * rango * np.sign(rng.random(d) - 0.5)
                dX_new[i] = 0.0

            X_new[i] = np.clip(X_new[i], lb, ub)

        X, dX = X_new, dX_new
        fit = _evaluar(f, X)
        n_eval += n_pop

        i_best, i_worst = int(np.argmin(fit)), int(np.argmax(fit))
        if fit[i_best] < food_f:
            food_f, food_x = fit[i_best], X[i_best].copy()
        if fit[i_worst] > enemy_f:
            enemy_f, enemy_x = fit[i_worst], X[i_worst].copy()

        curva.append(food_f)
        if verbose:
            print(f"[DA ] it {t:3d}  best = {food_f:.5f}")

    return {"best_x": food_x, "best_f": food_f, "curva": np.array(curva), "n_eval": n_eval}


# ---------------------------------------------------------------------------
# 2) Particle Swarm Optimization (PSO)
# ---------------------------------------------------------------------------
def pso(f, lb, ub, n_pop=20, max_iter=50, seed=None, w_max=0.9, w_min=0.4,
        c1=2.0, c2=2.0, vmax_frac=0.2, verbose=False):
    """
    PSO global con peso de inercia linealmente decreciente:
        v ← w·v + c1·r1·(pbest - x) + c2·r2·(gbest - x)
        x ← x + v
    La velocidad se limita a ±vmax_frac·(ub - lb).
    """
    rng = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    d = lb.size
    vmax = vmax_frac * (ub - lb)

    X = lb + rng.random((n_pop, d)) * (ub - lb)
    V = (rng.random((n_pop, d)) * 2 - 1) * vmax
    fit = _evaluar(f, X)
    n_eval = n_pop

    pbest, pbest_f = X.copy(), fit.copy()
    g = int(np.argmin(fit))
    gbest, gbest_f = X[g].copy(), fit[g]
    curva = [gbest_f]

    for t in range(1, max_iter + 1):
        w = w_max - (w_max - w_min) * t / max_iter
        r1, r2 = rng.random((n_pop, d)), rng.random((n_pop, d))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        V = np.clip(V, -vmax, vmax)
        X = np.clip(X + V, lb, ub)

        fit = _evaluar(f, X)
        n_eval += n_pop
        mejora = fit < pbest_f
        pbest[mejora], pbest_f[mejora] = X[mejora], fit[mejora]
        g = int(np.argmin(pbest_f))
        if pbest_f[g] < gbest_f:
            gbest, gbest_f = pbest[g].copy(), pbest_f[g]

        curva.append(gbest_f)
        if verbose:
            print(f"[PSO] it {t:3d}  best = {gbest_f:.5f}")

    return {"best_x": gbest, "best_f": gbest_f, "curva": np.array(curva), "n_eval": n_eval}


# ---------------------------------------------------------------------------
# 3) Differential Evolution (DE/rand/1/bin)
# ---------------------------------------------------------------------------
def differential_evolution(f, lb, ub, n_pop=20, max_iter=50, seed=None,
                           F=0.5, CR=0.9, verbose=False):
    """
    Evolución diferencial clásica DE/rand/1/bin:
        mutación : v = x_r1 + F·(x_r2 - x_r3)
        cruza    : u_j = v_j si rand < CR o j = j_rand, en otro caso x_j
        selección: x ← u si f(u) ≤ f(x)   (selección codiciosa)
    """
    rng = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    d = lb.size

    X = lb + rng.random((n_pop, d)) * (ub - lb)
    fit = _evaluar(f, X)
    n_eval = n_pop
    b = int(np.argmin(fit))
    curva = [fit[b]]

    for t in range(1, max_iter + 1):
        for i in range(n_pop):
            idx = [j for j in range(n_pop) if j != i]
            r1, r2, r3 = rng.choice(idx, 3, replace=False)
            v = X[r1] + F * (X[r2] - X[r3])
            v = np.clip(v, lb, ub)

            j_rand = rng.integers(d)
            mask = rng.random(d) < CR
            mask[j_rand] = True
            u = np.where(mask, v, X[i])

            fu = f(u)
            n_eval += 1
            if fu <= fit[i]:
                X[i], fit[i] = u, fu

        b = int(np.argmin(fit))
        curva.append(fit[b])
        if verbose:
            print(f"[DE ] it {t:3d}  best = {fit[b]:.5f}")

    return {"best_x": X[b].copy(), "best_f": fit[b], "curva": np.array(curva), "n_eval": n_eval}


ALGORITMOS = {
    "DA": dragonfly,
    "PSO": pso,
    "DE": differential_evolution,
}


if __name__ == "__main__":
    # Prueba rápida con la función esfera en 5D
    esfera = lambda x: float(np.sum(x ** 2))
    lb, ub = -5 * np.ones(5), 5 * np.ones(5)
    for nombre, alg in ALGORITMOS.items():
        r = alg(esfera, lb, ub, n_pop=20, max_iter=100, seed=0)
        print(f"{nombre:4s} best_f = {r['best_f']:.3e}  evals = {r['n_eval']}")

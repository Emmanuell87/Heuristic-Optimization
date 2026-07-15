# -*- coding: utf-8 -*-
"""Genera los GIFs animados de la Parte 1 (optimizacion numerica).
Salida: assets/parte1/*.gif  y  assets/parte1/*.png
Sin dependencia de scikit-opt: DG propio + PSO propio (con historia de enjambre).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

OUT = os.path.join("assets", "parte1")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------
# Funciones objetivo y gradientes
# ----------------------------------------------------------------------
def rosenbrock(x):
    x = np.asarray(x, dtype=float)
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

def grad_rosenbrock(x):
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    grad[:-1] += -400 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2 * (1 - x[:-1])
    grad[1:] += 200 * (x[1:] - x[:-1] ** 2)
    return grad

def rastrigin(x):
    x = np.asarray(x, dtype=float)
    A = 10
    n = len(x)
    return A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))

def grad_rastrigin(x):
    x = np.asarray(x, dtype=float)
    return 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)

limites_ros = (-2.048, 2.048)
limites_ras = (-5.12, 5.12)

# ----------------------------------------------------------------------
# Descenso por gradiente (con historia)
# ----------------------------------------------------------------------
def descenso_gradiente(func, grad_func, dim, limites, lr=0.001, max_iter=1000,
                       tol=1e-6, seed=None, x0=None):
    if seed is not None:
        np.random.seed(seed)
    x = np.array(x0, dtype=float) if x0 is not None else \
        np.random.uniform(limites[0], limites[1], dim)
    historia = [x.copy()]
    for i in range(max_iter):
        x_new = x - lr * grad_func(x)
        x_new = np.clip(x_new, limites[0], limites[1])
        historia.append(x_new.copy())
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x, func(x), np.array(historia), i + 1

# ----------------------------------------------------------------------
# PSO propio con historia completa del enjambre (para animar la poblacion)
# ----------------------------------------------------------------------
def pso(func, dim, limites, n_part=40, max_iter=60, w=0.72, c1=1.49, c2=1.49,
        seed=0):
    rng = np.random.default_rng(seed)
    lo, hi = limites
    X = rng.uniform(lo, hi, (n_part, dim))
    V = rng.uniform(-(hi - lo), (hi - lo), (n_part, dim)) * 0.1
    pbest = X.copy()
    pbest_val = np.array([func(p) for p in X])
    g_idx = int(np.argmin(pbest_val))
    gbest = pbest[g_idx].copy()
    gbest_val = pbest_val[g_idx]

    swarm_hist = [X.copy()]
    gbest_hist = [gbest.copy()]
    gbest_val_hist = [gbest_val]

    for _ in range(max_iter):
        r1 = rng.random((n_part, dim))
        r2 = rng.random((n_part, dim))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = np.clip(X + V, lo, hi)
        vals = np.array([func(p) for p in X])
        better = vals < pbest_val
        pbest[better] = X[better]
        pbest_val[better] = vals[better]
        g_idx = int(np.argmin(pbest_val))
        if pbest_val[g_idx] < gbest_val:
            gbest_val = pbest_val[g_idx]
            gbest = pbest[g_idx].copy()
        swarm_hist.append(X.copy())
        gbest_hist.append(gbest.copy())
        gbest_val_hist.append(gbest_val)

    return gbest, gbest_val, np.array(swarm_hist), np.array(gbest_hist), \
        np.array(gbest_val_hist)

# ----------------------------------------------------------------------
# Utilidades de malla / subsampleo
# ----------------------------------------------------------------------
def malla(func, limites, n=200):
    x = np.linspace(limites[0], limites[1], n)
    y = np.linspace(limites[0], limites[1], n)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[func(np.array([xi, yi])) for xi in x] for yi in y])
    return X, Y, Z

def subsample(hist, max_frames=70):
    if len(hist) <= max_frames:
        return hist, np.arange(len(hist))
    idx = np.linspace(0, len(hist) - 1, max_frames).astype(int)
    return hist[idx], idx

# ----------------------------------------------------------------------
# Animacion DG (una trayectoria)
# ----------------------------------------------------------------------
def anim_dg(func, historia, limites, titulo, out, fps=12):
    total = len(historia) - 1
    hist, idx = subsample(historia, 70)
    X, Y, Z = malla(func, limites)
    fig, ax = plt.subplots(figsize=(7, 6))
    cf = ax.contourf(X, Y, Z, levels=60, cmap="viridis")
    ax.contour(X, Y, Z, levels=15, colors="white", alpha=0.25, linewidths=0.5)
    fig.colorbar(cf, ax=ax, label="f(x, y)")
    linea, = ax.plot([], [], "-", color="#ff3b3b", lw=2.2)
    punto, = ax.plot([], [], "o", color="#ff3b3b", ms=9, mec="white")
    txt = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top",
                  color="white", fontsize=10,
                  bbox=dict(boxstyle="round", fc="black", alpha=0.5))
    ax.set_title(titulo, fontsize=12, fontweight="bold")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")

    def init():
        linea.set_data([], []); punto.set_data([], []); txt.set_text("")
        return linea, punto, txt

    def update(k):
        d = hist[:k + 1]
        linea.set_data(d[:, 0], d[:, 1])
        punto.set_data([d[-1, 0]], [d[-1, 1]])
        txt.set_text(f"iteracion {idx[k]}/{total}\nf = {func(d[-1]):.4f}")
        return linea, punto, txt

    an = FuncAnimation(fig, update, frames=len(hist), init_func=init,
                       blit=True, interval=1000 / fps)
    an.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print("  ->", out)

# ----------------------------------------------------------------------
# Animacion PSO (enjambre completo)
# ----------------------------------------------------------------------
def anim_pso(func, swarm_hist, gbest_hist, limites, titulo, out, fps=10):
    X, Y, Z = malla(func, limites)
    fig, ax = plt.subplots(figsize=(7, 6))
    cf = ax.contourf(X, Y, Z, levels=60, cmap="viridis")
    ax.contour(X, Y, Z, levels=15, colors="white", alpha=0.25, linewidths=0.5)
    fig.colorbar(cf, ax=ax, label="f(x, y)")
    part = ax.scatter([], [], s=42, c="#ffd23b", edgecolors="black",
                      linewidths=0.6, zorder=5)
    best, = ax.plot([], [], "*", color="#ff3b3b", ms=18, mec="white", zorder=6)
    txt = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top",
                  color="white", fontsize=10,
                  bbox=dict(boxstyle="round", fc="black", alpha=0.5))
    ax.set_title(titulo, fontsize=12, fontweight="bold")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")

    def init():
        part.set_offsets(np.empty((0, 2)))
        best.set_data([], []); txt.set_text("")
        return part, best, txt

    def update(k):
        part.set_offsets(swarm_hist[k])
        best.set_data([gbest_hist[k, 0]], [gbest_hist[k, 1]])
        txt.set_text(f"iter {k}/{len(swarm_hist)-1}\n"
                     f"f* = {func(gbest_hist[k]):.4f}")
        return part, best, txt

    an = FuncAnimation(fig, update, frames=len(swarm_hist), init_func=init,
                       blit=True, interval=1000 / fps)
    an.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print("  ->", out)

# ----------------------------------------------------------------------
# Superficie 3D estatica con trayectoria
# ----------------------------------------------------------------------
def surf3d(func, historia, limites, titulo, out):
    x = np.linspace(limites[0], limites[1], 80)
    y = np.linspace(limites[0], limites[1], 80)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[func(np.array([xi, yi])) for xi in x] for yi in y])
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.55, linewidth=0)
    zh = np.array([func(p) for p in historia])
    ax.plot(historia[:, 0], historia[:, 1], zh, "-", color="#ff3b3b", lw=2.5)
    ax.scatter(historia[-1, 0], historia[-1, 1], zh[-1], color="black", s=60)
    ax.set_title(titulo, fontsize=12, fontweight="bold")
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("f")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  ->", out)

# ======================================================================
# EJECUCION
# ======================================================================
print("== Descenso por Gradiente ==")
# Rosenbrock: lr pequeno, punto inicial que muestre el valle
_, _, h_ros_dg, it = descenso_gradiente(
    rosenbrock, grad_rosenbrock, 2, limites_ros, lr=0.0015,
    max_iter=1000, x0=[-1.8, 1.5])
print(f"DG Rosenbrock iters={it}, f={rosenbrock(h_ros_dg[-1]):.4f}")
anim_dg(rosenbrock, h_ros_dg, limites_ros,
        "Descenso por Gradiente — Rosenbrock", f"{OUT}/dg_rosenbrock.gif")
surf3d(rosenbrock, h_ros_dg, limites_ros,
       "Superficie 3D — Rosenbrock (DG)", f"{OUT}/dg_rosenbrock_3d.png")

# Rastrigin: se queda atrapado en un minimo local (evidencia el problema)
_, _, h_ras_dg, it = descenso_gradiente(
    rastrigin, grad_rastrigin, 2, limites_ras, lr=0.002,
    max_iter=400, x0=[3.7, -2.6])
print(f"DG Rastrigin iters={it}, f={rastrigin(h_ras_dg[-1]):.4f}")
anim_dg(rastrigin, h_ras_dg, limites_ras,
        "Descenso por Gradiente — Rastrigin (atrapado)",
        f"{OUT}/dg_rastrigin.gif")
surf3d(rastrigin, h_ras_dg, limites_ras,
       "Superficie 3D — Rastrigin (DG)", f"{OUT}/dg_rastrigin_3d.png")

print("== PSO ==")
gb, gv, sh, gh, gvh = pso(rosenbrock, 2, limites_ros, n_part=40,
                          max_iter=60, seed=1)
print(f"PSO Rosenbrock f*={gv:.6f} en {gb}")
anim_pso(rosenbrock, sh, gh, limites_ros,
         "PSO — Rosenbrock (enjambre)", f"{OUT}/pso_rosenbrock.gif")
surf3d(rosenbrock, gh, limites_ros,
       "Superficie 3D — Rosenbrock (PSO)", f"{OUT}/pso_rosenbrock_3d.png")

gb, gv, sh, gh, gvh = pso(rastrigin, 2, limites_ras, n_part=45,
                          max_iter=60, seed=3)
print(f"PSO Rastrigin f*={gv:.6f} en {gb}")
anim_pso(rastrigin, sh, gh, limites_ras,
         "PSO — Rastrigin (enjambre)", f"{OUT}/pso_rastrigin.gif")
surf3d(rastrigin, gh, limites_ras,
       "Superficie 3D — Rastrigin (PSO)", f"{OUT}/pso_rastrigin_3d.png")

print("LISTO parte 1")

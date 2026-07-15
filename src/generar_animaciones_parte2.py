# -*- coding: utf-8 -*-
"""Parte 2 (TSP Mexico): corre GA y ACO, genera mapas animados de la mejor
ruta evolucionando, curva de convergencia y guarda resultados/rutas a JSON.
Salida: assets/parte2/*.gif, *.png  y  assets/parte2/resultados.json
"""
import os, json, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from math import radians, sin, cos, atan2, sqrt

OUT = os.path.join("assets", "parte2")
os.makedirs(OUT, exist_ok=True)

# ---------------- Datos: 32 capitales ----------------
capitales = [
    ("Aguascalientes", 21.8853, -102.2916), ("Mexicali", 32.6245, -115.4523),
    ("La Paz", 24.1426, -110.3128), ("San Francisco de Campeche", 19.8450, -90.5230),
    ("Tuxtla Gutierrez", 16.7528, -93.1167), ("Chihuahua", 28.6329, -106.0691),
    ("Saltillo", 25.4267, -100.9954), ("Colima", 19.2433, -103.7250),
    ("Durango", 24.0277, -104.6532), ("Toluca", 19.2826, -99.6557),
    ("Guanajuato", 21.0190, -101.2574), ("Chilpancingo", 17.5515, -99.5006),
    ("Pachuca", 20.1011, -98.7591), ("Guadalajara", 20.6597, -103.3496),
    ("Morelia", 19.7060, -101.1950), ("Cuernavaca", 18.9242, -99.2216),
    ("Tepic", 21.5095, -104.8957), ("Monterrey", 25.6866, -100.3161),
    ("Oaxaca", 17.0732, -96.7266), ("Puebla", 19.0414, -98.2063),
    ("Queretaro", 20.5888, -100.3899), ("Chetumal", 18.5002, -88.2961),
    ("San Luis Potosi", 22.1565, -100.9855), ("Culiacan", 24.8091, -107.3940),
    ("Hermosillo", 29.0729, -110.9559), ("Villahermosa", 17.9892, -92.9475),
    ("Ciudad Victoria", 23.7369, -99.1411), ("Tlaxcala", 19.3182, -98.2375),
    ("Xalapa", 19.5438, -96.9102), ("Merida", 20.9674, -89.5926),
    ("Zacatecas", 22.7709, -102.5832), ("Ciudad de Mexico", 19.4326, -99.1332),
]
df = pd.DataFrame(capitales, columns=["ciudad", "lat", "lon"])
n = len(df)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

dist_km = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            dist_km[i, j] = haversine_km(df.loc[i,"lat"], df.loc[i,"lon"],
                                         df.loc[j,"lat"], df.loc[j,"lon"])

# ---------------- Parametros del modelo de costos ----------------
carro = {"nombre": "Nissan Versa 1.6", "rendimiento_km_l": 16.0}
precio_gasolina = 24.0        # MXN/L (Magna, promedio nacional 2025)
velocidad_prom_kmh = 85.0     # km/h carretera
factor_peaje_mxn_km = 0.85    # MXN/km (promedio red de cuota CAPUFE)
valores_hora = [100, 200, 300]  # MXN/h (analisis de sensibilidad)
VH_BASE = 200                 # caso base recomendado (ver justificacion)

def construir_matriz_costo(dist_km, vh, R, Pgas, v, fpeaje):
    costo_km = (Pgas / R) + (vh / v) + fpeaje
    C = dist_km * costo_km
    np.fill_diagonal(C, 0.0)
    return C, costo_km

def costo_ruta(ruta, C):
    total = 0.0
    for i in range(len(ruta)-1):
        total += C[ruta[i], ruta[i+1]]
    return total + C[ruta[-1], ruta[0]]

def distancia_ruta_km(ruta, D):
    total = 0.0
    for i in range(len(ruta)-1):
        total += D[ruta[i], ruta[i+1]]
    return total + D[ruta[-1], ruta[0]]

def nombres(ruta):
    return [df.loc[i, "ciudad"] for i in ruta]

# ---------------- GA (con snapshots de la mejor ruta) ----------------
def crossover_ox(p1, p2):
    n = len(p1); a, b = sorted(random.sample(range(n), 2))
    child = [-1]*n; child[a:b+1] = p1[a:b+1]
    fill = [x for x in p2 if x not in child]; idx = 0
    for i in range(n):
        if child[i] == -1: child[i] = fill[idx]; idx += 1
    return child

def mut_swap(route, p=0.2):
    r = route.copy()
    if random.random() < p:
        i, j = random.sample(range(len(r)), 2); r[i], r[j] = r[j], r[i]
    return r

def torneo(pob, fit, k=3):
    idxs = random.sample(range(len(pob)), k)
    return pob[min(idxs, key=lambda i: fit[i])]

def ga_tsp(C, pop_size=120, n_gen=450, p_mut=0.25, elite=5, seed=42, snaps=40):
    random.seed(seed); np.random.seed(seed)
    n = C.shape[0]
    pob = [random.sample(range(n), n) for _ in range(pop_size)]
    best_hist, route_snaps = [], []
    best_route, best_cost = None, np.inf
    snap_every = max(1, n_gen // snaps)
    for g in range(n_gen):
        fit = [costo_ruta(ind, C) for ind in pob]
        gi = int(np.argmin(fit))
        if fit[gi] < best_cost:
            best_cost = fit[gi]; best_route = pob[gi].copy()
        best_hist.append(best_cost)
        if g % snap_every == 0 or g == n_gen-1:
            route_snaps.append((g, best_route.copy(), best_cost))
        elite_idx = np.argsort(fit)[:elite]
        nueva = [pob[i].copy() for i in elite_idx]
        while len(nueva) < pop_size:
            h = crossover_ox(torneo(pob, fit), torneo(pob, fit))
            nueva.append(mut_swap(h, p=p_mut))
        pob = nueva
    return best_route, best_cost, best_hist, route_snaps

# ---------------- ACO (con snapshots) ----------------
def elegir_siguiente(act, novis, tau, eta, alpha, beta):
    cand = list(novis)
    probs = np.array([(tau[act,j]**alpha)*(eta[act,j]**beta) for j in cand])
    probs = probs / probs.sum()
    return int(np.random.choice(cand, p=probs))

def aco_tsp(C, n_ants=45, n_iter=280, alpha=1.0, beta=4.0, rho=0.45, Q=120.0,
            seed=42, snaps=40):
    np.random.seed(seed)
    n = C.shape[0]
    eta = np.where(C > 0, 1.0/(C + 1e-12), 0.0)
    tau = np.ones((n, n))
    best_route, best_cost = None, np.inf
    best_hist, route_snaps = [], []
    snap_every = max(1, n_iter // snaps)
    for it in range(n_iter):
        rutas, costos = [], []
        for _ in range(n_ants):
            start = np.random.randint(0, n)
            ruta = [start]; novis = set(range(n)); novis.remove(start)
            act = start
            while novis:
                nx = elegir_siguiente(act, novis, tau, eta, alpha, beta)
                ruta.append(nx); novis.remove(nx); act = nx
            c = costo_ruta(ruta, C); rutas.append(ruta); costos.append(c)
            if c < best_cost: best_cost = c; best_route = ruta.copy()
        tau *= (1 - rho)
        for ruta, c in zip(rutas, costos):
            dep = Q / (c + 1e-12)
            for i in range(len(ruta)-1):
                a, b = ruta[i], ruta[i+1]
                tau[a, b] += dep; tau[b, a] += dep
            tau[ruta[-1], ruta[0]] += dep; tau[ruta[0], ruta[-1]] += dep
        best_hist.append(best_cost)
        if it % snap_every == 0 or it == n_iter-1:
            route_snaps.append((it, best_route.copy(), best_cost))
    return best_route, best_cost, best_hist, route_snaps

# ---------------- Mapa animado de la mejor ruta evolucionando ----------------
def anim_ruta(route_snaps, titulo, out, color, fps=6):
    lons = df["lon"].values; lats = df["lat"].values
    fig, ax = plt.subplots(figsize=(7.5, 8.5))
    fig.patch.set_facecolor("#eef3f7")
    ax.set_facecolor("#f7fbff")
    ax.scatter(lons, lats, c="#333", s=22, zorder=4)
    for i in range(n):
        ax.text(lons[i], lats[i]+0.15, df.loc[i,"ciudad"], fontsize=5.5,
                ha="center", color="#444", zorder=5)
    linea, = ax.plot([], [], "-", color=color, lw=1.8, zorder=3)
    inicio, = ax.plot([], [], "o", color="#1b9e2f", ms=11, mec="white", zorder=6)
    titt = ax.set_title(titulo, fontsize=12, fontweight="bold")
    ax.set_xlabel("Longitud"); ax.set_ylabel("Latitud")
    ax.grid(alpha=0.25)
    txt = ax.text(0.02, 0.02, "", transform=ax.transAxes, va="bottom",
                  fontsize=10, bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    def frame(k):
        it, ruta, cost = route_snaps[k]
        seq = ruta + [ruta[0]]
        linea.set_data([lons[i] for i in seq], [lats[i] for i in seq])
        inicio.set_data([lons[ruta[0]]], [lats[ruta[0]]])
        txt.set_text(f"iter {it}   costo = {cost:,.0f} MXN")
        return linea, inicio, txt

    an = FuncAnimation(fig, frame, frames=len(route_snaps), blit=True,
                       interval=1000/fps)
    # repetir ultimo frame para que se aprecie la ruta final
    an.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print("  ->", out)

def plot_ruta_final(route, titulo, out, color):
    lons = df["lon"].values; lats = df["lat"].values
    seq = route + [route[0]]
    fig, ax = plt.subplots(figsize=(7.5, 8.5))
    ax.set_facecolor("#f7fbff")
    ax.plot([lons[i] for i in seq], [lats[i] for i in seq], "-", color=color, lw=1.9)
    ax.scatter(lons, lats, c="#333", s=22, zorder=4)
    ax.plot(lons[route[0]], lats[route[0]], "o", color="#1b9e2f", ms=12, mec="white", zorder=6)
    for i in range(n):
        ax.text(lons[i], lats[i]+0.15, df.loc[i,"ciudad"], fontsize=5.5, ha="center", color="#444")
    ax.set_title(titulo, fontsize=12, fontweight="bold")
    ax.set_xlabel("Longitud"); ax.set_ylabel("Latitud"); ax.grid(alpha=0.25)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("  ->", out)

# ======================================================================
print("== Corriendo GA y ACO para valores_hora ==")
resultados = []
conv = {}
for vh in valores_hora:
    C, ckm = construir_matriz_costo(dist_km, vh, carro["rendimiento_km_l"],
                                    precio_gasolina, velocidad_prom_kmh,
                                    factor_peaje_mxn_km)
    gr, gc, gh, gsnap = ga_tsp(C, seed=42+vh)
    ar, ac, ah, asnap = aco_tsp(C, seed=100+vh)
    resultados.append({"metodo":"GA","valor_hora":vh,"costo_total_mxn":round(gc,2),
                       "distancia_total_km":round(distancia_ruta_km(gr,dist_km),1),
                       "costo_prom_km_mxn":round(gc/distancia_ruta_km(gr,dist_km),3)})
    resultados.append({"metodo":"ACO","valor_hora":vh,"costo_total_mxn":round(ac,2),
                       "distancia_total_km":round(distancia_ruta_km(ar,dist_km),1),
                       "costo_prom_km_mxn":round(ac/distancia_ruta_km(ar,dist_km),3)})
    conv[vh] = {"GA":gh, "ACO":ah}
    if vh == VH_BASE:
        base = {"GA":(gr,gc,gsnap), "ACO":(ar,ac,asnap)}
    print(f"  vh={vh}: GA={gc:,.0f}  ACO={ac:,.0f}")

df_res = pd.DataFrame(resultados)
print(df_res.to_string(index=False))

# --- Elegir recomendacion en el caso base ---
gr, gc, gsnap = base["GA"]; ar, ac, asnap = base["ACO"]
if ac <= gc:
    rec_met, rec_route, rec_cost, rec_snap = "ACO", ar, ac, asnap
else:
    rec_met, rec_route, rec_cost, rec_snap = "GA", gr, gc, gsnap
# rotar para que inicie en Ciudad de Mexico
cdmx = df.index[df["ciudad"]=="Ciudad de Mexico"][0]
pos = rec_route.index(cdmx)
rec_route_cdmx = rec_route[pos:] + rec_route[:pos]

print(f"\nRECOMENDACION (vh={VH_BASE}): {rec_met}  costo={rec_cost:,.0f} MXN  "
      f"dist={distancia_ruta_km(rec_route,dist_km):,.0f} km")

# --- Animaciones de ruta (GA y ACO en el caso base) ---
print("== Animaciones de mapa ==")
anim_ruta(gsnap, f"Algoritmo Genetico — mejor ruta (vh={VH_BASE} MXN/h)",
          f"{OUT}/ga_ruta_mexico.gif", "#d62728")
anim_ruta(asnap, f"Colonia de Hormigas — mejor ruta (vh={VH_BASE} MXN/h)",
          f"{OUT}/aco_ruta_mexico.gif", "#1f77b4")
plot_ruta_final(rec_route_cdmx, f"Ruta recomendada ({rec_met}, vh={VH_BASE})",
                f"{OUT}/ruta_recomendada.png", "#6a3d9a")

# --- Convergencia GA vs ACO (caso base) ---
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(conv[VH_BASE]["GA"], lw=2, label="GA", color="#d62728")
ax.plot(conv[VH_BASE]["ACO"], lw=2, label="ACO", color="#1f77b4")
ax.set_title(f"Convergencia GA vs ACO (valor_hora={VH_BASE} MXN/h)", fontweight="bold")
ax.set_xlabel("Iteracion"); ax.set_ylabel("Mejor costo acumulado (MXN)")
ax.grid(alpha=0.3); ax.legend()
fig.savefig(f"{OUT}/convergencia_ga_vs_aco.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"  -> {OUT}/convergencia_ga_vs_aco.png")

# --- Guardar resultados y ruta recomendada ---
salida = {
    "carro": carro, "precio_gasolina_mxn_l": precio_gasolina,
    "velocidad_prom_kmh": velocidad_prom_kmh,
    "factor_peaje_mxn_km": factor_peaje_mxn_km,
    "valores_hora": valores_hora, "vh_base": VH_BASE,
    "costo_por_km_base": round((precio_gasolina/carro["rendimiento_km_l"])
                               + (VH_BASE/velocidad_prom_kmh) + factor_peaje_mxn_km, 3),
    "tabla": resultados,
    "recomendacion": {
        "metodo": rec_met, "valor_hora": VH_BASE,
        "costo_total_mxn": round(rec_cost, 2),
        "distancia_total_km": round(distancia_ruta_km(rec_route, dist_km), 1),
        "ruta_indices": rec_route_cdmx,
        "ruta_ciudades": nombres(rec_route_cdmx),
    },
}
with open(f"{OUT}/resultados.json", "w", encoding="utf-8") as f:
    json.dump(salida, f, ensure_ascii=False, indent=2)
print(f"  -> {OUT}/resultados.json")
print("LISTO parte 2")

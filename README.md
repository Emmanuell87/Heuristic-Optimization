# Optimización Heurística: Numérica y Combinatoria

**Trabajo 01 — Redes Neuronales y Algoritmos Bioinspirados**
Universidad Nacional de Colombia · Sede Medellín · Facultad de Minas
Docente: Juan David Ospina Arango

Análisis comparativo de métodos **deterministas** (descenso por gradiente) y
**heurísticos/bioinspirados** (algoritmos genéticos, PSO, evolución diferencial y
colonia de hormigas) sobre problemas de optimización numérica y combinatoria.

---

## 👥 Integrantes

| Integrante | Rol / Aportes principales |
|---|---|
| Jean Carlos Perilla García | Algoritmos evolutivos/heurísticos continuos (GA, PSO, DE) de la Parte 1; redacción y diseño del reporte |
| Emmanuel Alberto Mejía Arango | Modelado del TSP (Parte 2), coordenadas de las 32 capitales, matriz de costos; publicación del blog |
| Juan Camilo López Morales | Descenso por gradiente (Parte 1), colonia de hormigas (ACO) del TSP, scripts de exportación de animaciones (.gif) |

> Los aportes individuales detallados se encuentran en el video *"Reporte de
> contribución individual"* y en la sección de contribuciones del blog.

---

## 📝 Reporte técnico (blog)

El reporte completo está publicado como entrada de blog (contenido fuente en
[BLOG/Reporte_Técnico_de_Optimización_Numérica_y_Combina.md](Blog/Reporte_Técnico_de_Optimización_Numérica_y_Combina.md)):

**➡️ [Enlace al blog en Notion](https://app.notion.com/p/Reporte-T-cnico-de-Optimizaci-n-Num-rica-y-Combinatoria-39e959feddc680869fedd54a083b898d?source=copy_link)**

El video de contribución individual está disponible en:
**➡️ [Enlace al video](https://drive.google.com/file/d/1V6d8vGDEE92doS_9A4AMGjvxg_Kw6S6g/view?usp=drive_link)**

---

## 📂 Estructura del repositorio

```
Heuristic-Optimization/
├── Lab_Opt_Parte1_Rosenbrock_Rastrigin.ipynb   # Parte 1: optimización numérica
├── Parte_2_Optimización combinatoria.ipynb     # Parte 2: TSP México (GA + ACO)
├── BLOG.md                                      # Contenido del reporte/blog (fuente)
├── assets/
│   ├── parte1/                                  # GIFs y superficies 3D (numérica)
│   │   ├── dg_rosenbrock.gif   dg_rastrigin.gif
│   │   ├── pso_rosenbrock.gif  pso_rastrigin.gif
│   │   └── *_3d.png
│   └── parte2/                                  # Mapas animados y resultados (TSP)
│       ├── ga_ruta_mexico.gif  aco_ruta_mexico.gif
│       ├── ruta_recomendada.png  convergencia_ga_vs_aco.png
│       └── resultados.json
├── src/                                         # Scripts reproducibles de animaciones
│   ├── generar_animaciones_parte1.py
│   └── generar_animaciones_parte2.py
├── Reporte_Técnico_de_Optimización_Numérica_y_Combina.md            (Recomienda abrir el archivo en Notion)
└── README.md
```

---

## 🧮 Parte 1 — Optimización numérica

Se optimizan las funciones de **Rosenbrock** (valle estrecho, unimodal mal
condicionado) y **Rastrigin** (altamente multimodal) en **2D y 3D**, comparando:

- **Descenso por Gradiente (DG)** con condición inicial aleatoria.
- **Algoritmo Genético (GA)**, **PSO** y **Evolución Diferencial (DE)**.

Cada método registra el **valor final de la función objetivo** y el **número de
evaluaciones**, con **30 corridas independientes** para dar validez estadística.

| Método | Valor final (≈) | Evaluaciones | Aporte |
|---|---|---|---|
| Descenso por Gradiente | 0.99 (Ros) / 18–20 (Ras) | ~30–1000 | Explotación local, sensible al inicio |
| Algoritmo Genético | 0.07 / 0.69 | ~10 000 | Buena exploración, precisión final limitada |
| PSO | ≈0.0 / ≈0.0 | ~6 000 | Robusto y estable |
| Evolución Diferencial | ≈0.0 / ≈0.0 | ~2 000–4 000 | Mejor balance precisión/eficiencia |

## 🗺️ Parte 2 — Optimización combinatoria (TSP México)

Recorrido óptimo por las **32 capitales estatales de México** minimizando un costo
de desplazamiento realista:

```
Costo(i→j) = combustible + peaje + tiempo
           = (d_ij / R)·P_gas  +  d_ij·f_peaje  +  d_ij·(V_h / v)
```

Se resuelve con **Algoritmo Genético (GA)** y **Colonia de Hormigas (ACO)**.
Vehículo del vendedor: **Nissan Versa 1.6** (≈16 km/L). En todos los escenarios
**ACO** obtuvo la mejor ruta; la ruta recomendada (caso base `V_h = 200 MXN/h`)
tiene un costo de **≈41 533 MXN** y **≈8 831 km**.

---

## ▶️ Reproducir los resultados

Requisitos:

```bash
pip install numpy pandas matplotlib scipy pillow scikit-opt
```

Ejecutar los notebooks de arriba hacia abajo:

- `Lab_Opt_Parte1_Rosenbrock_Rastrigin.ipynb`
- `Parte_2_Optimización combinatoria.ipynb`

Los GIFs y figuras se guardan automáticamente en `assets/`.

---

## 🤖 Uso de IA

El desarrollo se apoyó en asistentes de IA para estructuración de código,
depuración, diseño experimental y redacción. Los prompts principales y su impacto
se documentan en la sección *"Uso de IA"* del blog. Todas las decisiones
metodológicas, la interpretación de resultados y la validación fueron
responsabilidad del equipo.

---

## 📚 Bibliografía

Referencias completas en normas **APA** en el blog. Fuentes base: Nocedal & Wright
(2006), Kennedy & Eberhart (1995), Storn & Price (1997), Dorigo & Stützle (2004),
Goldberg (1989), entre otras.

<!--
============================================================================
 BLOG FINAL (FUSIONADO) — LISTO PARA PEGAR EN NOTION
============================================================================
 Fusiona la Parte 1 (redacción de Jean/equipo) + la Parte 2 completa + Uso de IA
 + aportes + bibliografía unificada. Los números de las figuras ya coinciden con
 los GIFs de la carpeta assets/.

 CÓMO USARLO:
 1. Pega el contenido en Notion (bloque por bloque o todo).
 2. Donde diga  [[ IMAGEN: assets/... ]]  sube ESE archivo como bloque de imagen
    y debajo deja el pie de figura ya redactado.
 3. Las ecuaciones van entre  $$ ... $$  → Notion las convierte a KaTeX.
============================================================================
-->

# Optimización Numérica y Combinatoria: Análisis Comparativo de Métodos Deterministas y Heurísticos

**Curso:** Redes Neuronales y Algoritmos Bioinspirados
**Docente:** Juan David Ospina Arango
**Institución:** Universidad Nacional de Colombia, Sede Medellín — Facultad de Minas
**Repositorio de código:** https://github.com/Emmanuell87/Heuristic-Optimization

## Integrantes e información de coautoría

- **Jean Carlos Perilla García** — Desarrollo y calibración de los algoritmos evolutivos y heurísticos continuos (GA, PSO, DE) de la Parte 1; redacción y diseño del reporte técnico.
- **Emmanuel Alberto Mejía Arango** — Modelado matemático del problema combinatorio (TSP México), recolección de coordenadas geográficas de las 32 capitales y estructuración de la matriz de costos de la Parte 2; publicación del blog.
- **Juan Camilo López Morales** — Implementación del descenso por gradiente de la Parte 1, codificación de la colonia de hormigas (ACO) para el TSP y generación de los scripts de exportación de animaciones (.gif).

---

# PARTE 1: Optimización Numérica

## 1. Introducción y selección de funciones de prueba

En el diseño de sistemas de ingeniería y la modelación matemática, la optimización de funciones continuas representa un desafío fundamental. La topología y el paisaje de búsqueda (*search landscape*) de un problema determinan directamente el éxito o fracaso de un algoritmo. Mientras que los espacios convexos y suaves facilitan el trabajo de los métodos tradicionales, las superficies rugosas y multimodales exigen estrategias de exploración global (Jamil & Yang, 2013).

Para evaluar la robustez, límites y eficiencia de los enfoques deterministas frente a los metaheurísticos, se seleccionaron dos funciones de prueba (*benchmarks*) que representan dos mundos geométricos opuestos: la función de Rosenbrock y la función de Rastrigin.

### 1.1 Función de Rosenbrock (el valle no convexo)

La función de Rosenbrock, propuesta por Rosenbrock (1960) y conocida como *función del valle banana*, es una superficie no convexa ampliamente usada para evaluar algoritmos de optimización continua. Su formulación general para $n$ dimensiones es la **Ecuación 1**:

$$
f(x)=\sum_{i=1}^{n-1}[100(x_{i+1}-x_{i}^{2})^{2}+(1-x_{i})^{2}] \quad \textbf{(Ecuación 1)}
$$

Presenta un único mínimo global en $x^*=(1,1,\dots,1)$, donde $f(x^*)=0$.

**Geometría y dificultad.** Su complicación no radica en múltiples mínimos locales, sino en su topología: un valle estrecho, curvo, asimétrico y alargado. Al ingresar al valle, los métodos basados en gradiente reducen drásticamente la magnitud de sus pasos; como las paredes son empinadas y el fondo casi plano, sufren oscilaciones y convergencia lenta, muy sensibles a la tasa de aprendizaje (Nocedal & Wright, 2006).

**Justificación.** Es ideal para evaluar la explotación local y el ajuste de paso: mide cómo un método navega valles mal condicionados siguiendo trayectorias no lineales sin perder estabilidad.

### 1.2 Función de Rastrigin (la trampa multimodal)

La función de Rastrigin (Rastrigin, 1974; Mühlenbein et al., 1991) es altamente multimodal. Su formulación en $n$ dimensiones es la **Ecuación 2**:

$$
f(x)=10n+\sum_{i=1}^{n}[x_{i}^{2}-10\cos(2\pi x_{i})] \quad \textbf{(Ecuación 2)}
$$

Su mínimo global está en el origen $x^*=(0,0,\dots,0)$ con $f(x^*)=0$.

**Geometría y dificultad.** Combina una parábola global con un término cosenoidal periódico, generando una superficie rugosa densamente poblada de mínimos locales. Todo algoritmo que dependa de información local es atraído al mínimo local más cercano, sufriendo convergencia prematura.

**Justificación.** Pone a prueba la exploración global de las metaheurísticas: mide la robustez para sobrevolar crestas oscilatorias y evitar el estancamiento local.

## 2. Optimización tradicional: descenso por gradiente

### 2.1 Fundamento teórico

El descenso por gradiente es un algoritmo determinista de primer orden para hallar mínimos locales de funciones continuas y diferenciables. Calcula iterativamente el gradiente en la posición actual y se desplaza en la dirección opuesta (máximo descenso local). La regla de actualización es la **Ecuación 3**:

$$
x^{(k+1)}=x^{(k)}-\eta\nabla f(x^{(k)}) \quad \textbf{(Ecuación 3)}
$$

donde $x^{(k)}$ es el vector de variables en la iteración $k$, $\eta\in\mathbb{R}^+$ es la tasa de aprendizaje y $\nabla f(x^{(k)})$ el gradiente. El proceso se repite hasta alcanzar el máximo de iteraciones, una norma del gradiente menor que una tolerancia ($\lVert\nabla f(x)\rVert<\text{tol}$), o un cambio mínimo en la función (Nocedal & Wright, 2006).

### 2.2 Gradientes analíticos en 2D y 3D

Para direcciones de descenso exactas se dedujeron las derivadas parciales de cada función.

**Rosenbrock, gradiente 2D** con $f(x,y)=100(y-x^2)^2+(1-x)^2$ (**Ecuación 4**):

$$
\nabla f(x,y)=\begin{bmatrix} -400x(y-x^{2})-2(1-x) \\ 200(y-x^{2}) \end{bmatrix} \quad \textbf{(Ecuación 4)}
$$

**Rosenbrock, gradiente 3D** (acoplamiento en cadena de variables) (**Ecuación 5**):

$$
\nabla f(x,y,z)=\begin{bmatrix} -400x(y-x^{2})-2(1-x) \\ 200(y-x^{2})-400y(z-y^{2})-2(1-y) \\ 200(z-y^{2}) \end{bmatrix} \quad \textbf{(Ecuación 5)}
$$

**Rastrigin, derivada por componente** (**Ecuación 6**):

$$
\frac{\partial f}{\partial x_{i}}=2x_{i}+20\pi\sin(2\pi x_{i}) \quad \textbf{(Ecuación 6)}
$$

de donde los gradientes 2D y 3D son las **Ecuaciones 7 y 8**:

$$
\nabla f(x,y)=\begin{bmatrix} 2x+20\pi\sin(2\pi x) \\ 2y+20\pi\sin(2\pi y) \end{bmatrix} \quad \textbf{(Ecuación 7)}
$$

$$
\nabla f(x,y,z)=\begin{bmatrix} 2x+20\pi\sin(2\pi x) \\ 2y+20\pi\sin(2\pi y) \\ 2z+20\pi\sin(2\pi z) \end{bmatrix} \quad \textbf{(Ecuación 8)}
$$

### 2.3 Visualización dinámica de las trayectorias (2D)

> *Las animaciones se exportaron con el módulo `PillowWriter` de Matplotlib y se guardaron en el repositorio para garantizar su renderizado dinámico en el blog.*

#### 2.3.1 Descenso por gradiente en Rosenbrock (2D)

Espacio de búsqueda $x,y\in[-2.048,\,2.048]$, con condición inicial aleatoria.

> [[ IMAGEN: assets/parte1/dg_rosenbrock.gif ]]
> **Figura 1.** Evolución iterativa del descenso por gradiente navegando el valle de Rosenbrock.

**Análisis métrico.** El algoritmo recorrió el valle y tras agotar sus **1000 iteraciones** terminó en $x\approx[0.889,\,0.790]$, con un valor final $f(x)\approx 0.0123$. Se acercó mucho al óptimo $(1,1)$ pero **no alcanzó el cero exacto**: el punto desciende rápido por las laderas empinadas y luego avanza con lentitud por el fondo curvo y casi plano del valle, confirmando el comportamiento teórico esperado.

#### 2.3.2 Descenso por gradiente en Rastrigin (2D)

Dominio $x,y\in[-5.12,\,5.12]$.

> [[ IMAGEN: assets/parte1/dg_rastrigin.gif ]]
> **Figura 2.** Descenso por gradiente en el paisaje rugoso de Rastrigin: el algoritmo queda atrapado.

**Análisis métrico.** En esta superficie multimodal el algoritmo se detuvo de forma **prematura tras solo 12 iteraciones** (norma del gradiente por debajo de la tolerancia), quedando confinado en el mínimo local $x\approx[3.98,\,-2.98]$ con $f(x)\approx 24.87$, muy lejos del óptimo global en el origen. Esta es una corrida individual representativa; el promedio de 30 corridas (Tabla 1) es $18.97\pm10.08$, y este resultado cae dentro de esa dispersión, evidenciando que el desempeño depende del azar de la semilla inicial.

## 3. Optimización metaheurística: enfoque bioinspirado y poblacional

### 3.1 Justificación del enfoque global

A diferencia de las técnicas deterministas, las metaheurísticas no dependen de la información local del gradiente: operan sobre una **población** de soluciones distribuidas en el dominio. Al incorporar operadores estocásticos y mecanismos de comunicación interna, equilibran la exploración global con la explotación de las mejores regiones (Eiben & Smith, 2015), lo que las hace idóneas para topologías no convexas (Rosenbrock) o paisajes rugosos (Rastrigin).

### 3.2 Algoritmos evaluados

**Algoritmo Genético (GA).** Inspirado en la selección natural (Holland, 1975; Goldberg, 1989). Población de 50 individuos, 200 generaciones, selección por torneo ($k=3$), cruce por orden (OX), mutación por intercambio ($P_{\text{mut}}=0.25$) y elitismo (5 mejores). Buena exploración global, pero con dificultad para la sintonía fina decimal cerca del óptimo continuo.

**Enjambre de Partículas (PSO).** Basado en el comportamiento social de bandadas (Kennedy & Eberhart, 1995). Cada partícula ajusta su velocidad según su memoria y la del enjambre; la posición se actualiza según la **Ecuación 9**:

$$
x_{i}^{(k+1)}=x_{i}^{(k)}+v_{i}^{(k+1)} \quad \textbf{(Ecuación 9)}
$$

Enjambre de 40 partículas, 150 iteraciones. Convergencia acelerada y estable; sobrevuela las crestas de Rastrigin sin quedar atrapado.

**Evolución Diferencial (DE).** Metaheurística poblacional (Storn & Price, 1997) que genera vectores mutantes combinando diferencias de individuos, según la **Ecuación 10**:

$$
v_{i}^{(k+1)}=x_{r_1}^{(k)}+F\cdot(x_{r_2}^{(k)}-x_{r_3}^{(k)}) \quad \textbf{(Ecuación 10)}
$$

con $r_1,r_2,r_3$ índices aleatorios distintos y $F\in[0,2]$ el factor de escala. Estrategia `best1bin` (SciPy), `popsize=15`. El algoritmo más preciso y eficiente del estudio.

### 3.3 Visualización dinámica de las poblaciones (2D)

#### 3.3.1 PSO en Rosenbrock (2D)

> [[ IMAGEN: assets/parte1/pso_rosenbrock.gif ]]
> **Figura 3.** Movimiento del enjambre (PSO) en Rosenbrock. Las partículas de las paredes de alta energía descienden y guían al grupo por el pasillo curvo hacia el mínimo; la mejor partícula alcanzó $f\approx 2.7\times10^{-5}$ en $x\approx[0.997,\,0.993]$.

#### 3.3.2 PSO en Rastrigin (2D)

> [[ IMAGEN: assets/parte1/pso_rastrigin.gif ]]
> **Figura 4.** Dinámica exploratoria del enjambre (PSO) en Rastrigin. A diferencia de la Figura 2, el enjambre sobrevuela los mínimos locales y concentra la búsqueda en el centro del dominio, alcanzando $f\approx 1.3\times10^{-5}$ en $x\approx[0,\,0]$.

#### 3.3.3 Superficies 3D

> [[ IMAGEN: assets/parte1/dg_rosenbrock_3d.png ]]  ·  [[ IMAGEN: assets/parte1/pso_rastrigin_3d.png ]]
> **Figura 5.** Vistas 3D de la superficie objetivo con la trayectoria proyectada: descenso por gradiente sobre Rosenbrock (izq.) y enjambre PSO sobre Rastrigin (der.).

## 4. Discusión comparativa y análisis estadístico

Dado el componente estocástico de los algoritmos, una sola ejecución carece de validez. Se ejecutaron **30 corridas independientes** por método sobre las funciones en 2D; los resultados se consolidan en la **Tabla 1**.

**Tabla 1.** Análisis estadístico comparativo de desempeño (30 ejecuciones independientes).

| Función / Algoritmo | Descenso por Gradiente (DG) | Algoritmo Genético (GA) | Enjambre de Partículas (PSO) | Evolución Diferencial (DE) |
|---|---|---|---|---|
| **Rosenbrock** (valor final) | $0.9921 \pm 0.4120$ | $0.0712 \pm 0.0150$ | **0.000000 ± 0.0** | **0.000000 ± 0.0** |
| Evaluaciones promedio | 1 000 *(límite)* | 10 001 | 6 040 | **3 990** |
| **Rastrigin** (valor final) | $18.9741 \pm 10.0810$ | $0.6921 \pm 0.1240$ | **0.000000 ± 0.0** | **0.000000 ± 0.0** |
| Evaluaciones promedio | **27.9** | 10 001 | 6 040 | **1 987** |

*Nota: en negrita el mejor desempeño por métrica. Las evaluaciones heurísticas incluyen el costo acumulado de la población.*

**Descenso por gradiente.** Altísima eficiencia local (27.9 evaluaciones en Rastrigin), pero ese bajo número no es éxito sino **falla por convergencia prematura**: queda atrapado en mínimos locales con valor medio deficiente ($18.97$) y desviación enorme ($\pm10.08$). En Rosenbrock agota sus 1000 iteraciones rebotando en el valle sin sintonía fina.

**Metaheurísticas.** Justifican el mayor número de evaluaciones al evadir trampas locales. **GA** se aproxima al óptimo pero con precisión decimal limitada ($0.07$ y $0.69$). **PSO** converge al cero con estabilidad absoluta ($\pm0.0$) a costo constante (6 040 evaluaciones). **DE** es la técnica superior: alcanza el óptimo perfecto con el menor costo (**3 990** en Rosenbrock, **1 987** en Rastrigin) gracias a su parada dinámica. Un mayor número de evaluaciones no es desperdicio, sino el costo justificado para garantizar convergencia global en entornos no convexos y multimodales.

---

# PARTE 2: Optimización Combinatoria (TSP México)

## 5. Planteamiento

Un vendedor debe recorrer las **32 capitales estatales de México** y regresar al origen minimizando el costo total de desplazamiento. Es una instancia del problema del vendedor viajero (TSP), NP-difícil, para el cual las metaheurísticas son la vía práctica en instancias medianas y grandes (Lawler et al., 1985; Reinelt, 1994). Se resuelve con **Algoritmo Genético (GA)** y **Colonia de Hormigas (ACO)**.

## 6. Datos y construcción de la matriz de costos

> **Premisas, fuentes y procedimiento** (atendiendo la observación del docente).

### 6.1 Coordenadas y distancias

Se construyeron 32 nodos con las coordenadas geográficas (lat, lon) de cada capital, tomadas de registros públicos de geolocalización (INEGI). La distancia entre ciudades se aproximó con la **fórmula de Haversine** (Sinnott, 1984), distancia de círculo máximo sobre la esfera terrestre (**Ecuación 11**):

$$
d_{ij}=2R\,\arcsin\sqrt{\sin^{2}\!\left(\tfrac{\Delta\phi}{2}\right)+\cos\phi_i\cos\phi_j\sin^{2}\!\left(\tfrac{\Delta\lambda}{2}\right)} \quad \textbf{(Ecuación 11)}
$$

con $R=6371$ km. Es adecuada sin una API de rutas por carretera, aunque **subestima** las distancias reales (limitación discutida en la Sección 9).

### 6.2 Modelo de costo por tramo

El costo de ir de $i$ a $j$ suma tres componentes (**Ecuación 12**):

$$
C_{ij}=C^{\text{comb}}_{ij}+C^{\text{peaje}}_{ij}+C^{\text{tiempo}}_{ij} \quad \textbf{(Ecuación 12)}
$$

$$
C^{\text{comb}}_{ij}=d_{ij}\frac{P_{\text{gas}}}{R_{\text{veh}}},\qquad
C^{\text{peaje}}_{ij}=d_{ij}\,f_{\text{peaje}},\qquad
C^{\text{tiempo}}_{ij}=d_{ij}\frac{V_h}{v} \quad \textbf{(Ecuación 13)}
$$

El costo total de una ruta $\pi$ incluye el retorno al origen (**Ecuación 14**):

$$
C_{\text{total}}(\pi)=\sum_{k=1}^{n-1}C_{\pi_k\pi_{k+1}}+C_{\pi_n\pi_1} \quad \textbf{(Ecuación 14)}
$$

### 6.3 Parámetros: valores, fuentes y justificación

**Vehículo del vendedor:** **Nissan Versa 1.6**, sedán compacto muy común en México, con rendimiento en carretera $R_{\text{veh}}\approx16$ km/L (CONUEE *ecovehículos* / fabricante).

**Tabla 2.** Premisas del modelo de costos y su fuente.

| Parámetro | Símbolo | Valor | Fuente / justificación |
|---|---|---|---|
| Precio gasolina Magna | $P_{\text{gas}}$ | 24.0 MXN/L | Promedio nacional 2025 — Comisión Reguladora de Energía (CRE) / Pemex |
| Rendimiento del vehículo | $R_{\text{veh}}$ | 16.0 km/L | Nissan Versa 1.6 en carretera (CONUEE / fabricante) |
| Factor de peaje | $f_{\text{peaje}}$ | 0.85 MXN/km | Costo medio de la red de autopistas de cuota (CAPUFE / SICT) |
| Velocidad promedio | $v$ | 85 km/h | Velocidad media realista en carretera federal (SICT) |
| Valor de la hora | $V_h$ | 100 / 200 / 300 MXN/h | Variable de decisión; anclada al salario mínimo (CONASAMI, 2025). Caso base: 200 |

**Sobre $V_h$.** El salario mínimo general 2025 es ≈ 278.80 MXN/día (CONASAMI, 2025); un vendedor calificado con vehículo tiene un valor por hora superior. Al haber incertidumbre, se trató como **variable de decisión** con un **análisis de sensibilidad** en $\{100,200,300\}$ MXN/h; se adoptó $V_h=200$ como caso base.

**Costo por kilómetro (caso base $V_h=200$)** (**Ecuación 15**):

$$
c_{\text{km}}=\underbrace{\tfrac{24}{16}}_{1.50}+\underbrace{0.85}_{\text{peaje}}+\underbrace{\tfrac{200}{85}}_{2.35}=4.70\ \text{MXN/km} \quad \textbf{(Ecuación 15)}
$$

## 7. Algoritmos implementados

- **Algoritmo Genético (GA):** población de permutaciones, selección por torneo, cruce por orden (OX) y mutación por intercambio con elitismo (Goldberg, 1989; Larrañaga et al., 1999).
- **Colonia de Hormigas (ACO):** construcción probabilística de rutas guiada por feromona $\tau$ y visibilidad $\eta=1/C_{ij}$, con evaporación y depósito proporcional a la calidad (Dorigo & Stützle, 2004). La probabilidad de transición es la **Ecuación 16**:

$$
p_{ij}=\frac{\tau_{ij}^{\alpha}\,\eta_{ij}^{\beta}}{\sum_{l\in\mathcal{N}}\tau_{il}^{\alpha}\,\eta_{il}^{\beta}} \quad \textbf{(Ecuación 16)}
$$

## 8. Resultados

### 8.1 Desempeño GA vs ACO y análisis de sensibilidad

**Tabla 3.** Costo total y distancia de la mejor ruta por método y valor de la hora.

| Método | $V_h$ (MXN/h) | Costo total (MXN) | Distancia (km) | Costo/km (MXN) |
|---|---|---|---|---|
| ACO | 100 | **32 357** | 9 176 | 3.53 |
| GA  | 100 | 34 693 | 9 838 | 3.53 |
| ACO | 200 | **41 533** | 8 831 | 4.70 |
| GA  | 200 | 50 602 | 10 760 | 4.70 |
| ACO | 300 | **53 169** | 9 043 | 5.88 |
| GA  | 300 | 55 387 | 9 421 | 5.88 |

**ACO supera a GA en los tres escenarios.** Un hallazgo relevante: el **costo/km es idéntico para ambos métodos en cada $V_h$** porque $c_{\text{km}}$ es constante, de modo que $C_{\text{total}}=c_{\text{km}}\cdot D_{\text{total}}$. Por tanto **minimizar costo equivale a minimizar distancia**, y el valor de la hora **escala** el costo total sin alterar el orden óptimo de visita (se retoma como limitación en la Sección 9).

### 8.2 Convergencia

> [[ IMAGEN: assets/parte2/convergencia_ga_vs_aco.png ]]
> **Figura 6.** Evolución del mejor costo acumulado por iteración (caso base $V_h=200$). ACO reduce el costo más rápido y estabiliza en un valor menor que GA, coherente con su mejor explotación de la feromona.

### 8.3 Visualización geográfica animada

> [[ IMAGEN: assets/parte2/ga_ruta_mexico.gif ]]
> **Figura 7.** Evolución de la mejor ruta del **Algoritmo Genético** a lo largo de las generaciones (caso base). La ruta pasa de un trazado enredado a uno ordenado, terminando en ≈ 50 602 MXN.

> [[ IMAGEN: assets/parte2/aco_ruta_mexico.gif ]]
> **Figura 8.** Evolución de la mejor ruta de la **Colonia de Hormigas** (caso base). ACO alcanza una ruta más limpia y de menor costo (≈ 41 533 MXN) que GA.

> [[ IMAGEN: assets/parte2/ruta_recomendada.png ]]
> **Figura 9.** Ruta recomendada al vendedor (ACO, $V_h=200$), iniciando y terminando en Ciudad de México.

### 8.4 Ruta recomendada al vendedor y justificación

**Se recomienda la ruta obtenida por ACO** (caso base $V_h=200$ MXN/h), con **costo ≈ 41 533 MXN** y **≈ 8 831 km**, por tres razones:

1. **Menor costo y distancia:** es la ruta de menor distancia hallada (8 831 km) y, al ser el costo proporcional a la distancia, también la de menor costo operativo.
2. **Robustez del método:** ACO superó a GA en los tres valores de la hora (Tabla 3) y convergió de forma más estable (Figura 6).
3. **Coherencia geográfica:** agrupa regiones (Bajío, occidente, noroeste, noreste, sureste y sur) minimizando cruces (Figura 9).

**Orden de visita recomendado** (ciclo; el punto de inicio es intercambiable):

> Ciudad de México → Toluca → Morelia → Colima → Guadalajara → Tepic → Aguascalientes → Zacatecas → Durango → Culiacán → La Paz → Hermosillo → Mexicali → Chihuahua → Saltillo → Monterrey → Ciudad Victoria → San Luis Potosí → Guanajuato → Querétaro → Pachuca → Tlaxcala → Puebla → Xalapa → Chetumal → Mérida → San Francisco de Campeche → Villahermosa → Tuxtla Gutiérrez → Oaxaca → Chilpancingo → Cuernavaca → (regreso a Ciudad de México)

## 9. Discusión y limitaciones

- **GA y ACO** son apropiados para el TSP por su búsqueda global en espacios combinatorios grandes (Dorigo & Stützle, 2004).
- **Distancia Haversine:** simplifica el modelado, pero subestima distancias reales por carretera. Ejemplo: el tramo Culiacán–La Paz cruza el Golfo de California en línea recta, cuando por carretera La Paz solo se conecta vía Baja California; una API vial (OSRM, Google Directions) corregiría estos casos.
- **Modelo de costo lineal:** al ser $c_{\text{km}}$ constante, el valor de la hora no cambia la ruta óptima. Incorporar **peajes reales por tramo** (tarifas CAPUFE por caseta) rompería la proporcionalidad y haría que $V_h$ sí modifique la ruta.
- **Trabajo futuro:** integrar distancias y peajes reales por carretera y comparar con soluciones exactas o de referencia (LKH) para medir la brecha de optimalidad.

## 10. Conclusiones

1. Los métodos heurísticos superaron al descenso por gradiente ante la multimodalidad, alcanzando el óptimo global donde el gradiente quedó atrapado.
2. En la Parte 1, **DE** ofreció el mejor balance precisión/eficiencia, seguido de PSO; GA exploró bien pero con menor precisión final.
3. En la Parte 2, **ACO** dominó a GA en costo y estabilidad; se recomienda su ruta.
4. Las animaciones permitieron validar experimentalmente el comportamiento teórico de cada algoritmo.

---

## 11. Uso de IA (obligatorio)

Se emplearon asistentes de IA como apoyo de productividad. Prompts principales:

1. *"Implementa en Python el descenso por gradiente para Rosenbrock y Rastrigin en 2D y 3D, devolviendo el punto óptimo y el historial de posiciones para animar."*
2. *"Optimiza las mismas funciones con GA, PSO y evolución diferencial usando librerías científicas y registrando el número de evaluaciones."*
3. *"Genera funciones con Matplotlib para animar la trayectoria en contornos 2D y mostrar la superficie 3D con la trayectoria."*
4. *"Modela un TSP con costo compuesto por combustible, peajes y tiempo, y resuélvelo con GA y ACO para 32 ciudades; anima la mejor ruta sobre el mapa."*
5. *"Redacta la estructura del reporte técnico con metodología, discusión y bibliografía en APA."*

**Impacto.** La IA aceleró la estructuración del código, la depuración, el diseño experimental y la redacción. Sin embargo, el equipo **validó y ajustó** parámetros, detectó y corrigió comportamientos (p. ej., la inestabilidad del gradiente y el artefacto de la distancia Haversine), interpretó los resultados y tomó todas las decisiones metodológicas. La IA funcionó como asistente, no como autor de las conclusiones.

---

## 12. Referencias (APA, 7.ª edición)

Comisión Nacional de los Salarios Mínimos (CONASAMI). (2025). *Tabla de salarios mínimos generales y profesionales*. Gobierno de México.

Dorigo, M., & Stützle, T. (2004). *Ant colony optimization*. MIT Press.

Eiben, A. E., & Smith, J. E. (2015). *Introduction to evolutionary computing* (2nd ed.). Springer.

Goldberg, D. E. (1989). *Genetic algorithms in search, optimization, and machine learning*. Addison-Wesley.

Holland, J. H. (1975). *Adaptation in natural and artificial systems*. University of Michigan Press.

Jamil, M., & Yang, X.-S. (2013). A literature survey of benchmark functions for global optimisation problems. *International Journal of Mathematical Modelling and Numerical Optimisation, 4*(2), 150–194.

Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. En *Proceedings of the IEEE International Conference on Neural Networks* (pp. 1942–1948). IEEE.

Larrañaga, P., Kuijpers, C. M. H., Murga, R. H., Inza, I., & Dizdarevic, S. (1999). Genetic algorithms for the travelling salesman problem: A review of representations and operators. *Artificial Intelligence Review, 13*(2), 129–170.

Lawler, E. L., Lenstra, J. K., Rinnooy Kan, A. H. G., & Shmoys, D. B. (Eds.). (1985). *The traveling salesman problem: A guided tour of combinatorial optimization*. Wiley.

Mühlenbein, H., Schomisch, M., & Born, J. (1991). The parallel genetic algorithm as function optimizer. *Parallel Computing, 17*(6–7), 619–632.

Nocedal, J., & Wright, S. J. (2006). *Numerical optimization* (2nd ed.). Springer.

Rastrigin, L. A. (1974). *Systems of extremal control*. Nauka.

Reinelt, G. (1994). *The traveling salesman: Computational solutions for TSP applications*. Springer.

Rosenbrock, H. H. (1960). An automatic method for finding the greatest or least value of a function. *The Computer Journal, 3*(3), 175–184.

Sinnott, R. W. (1984). Virtues of the Haversine. *Sky and Telescope, 68*(2), 159.

Storn, R., & Price, K. (1997). Differential evolution — A simple and efficient heuristic for global optimization over continuous spaces. *Journal of Global Optimization, 11*(4), 341–359.

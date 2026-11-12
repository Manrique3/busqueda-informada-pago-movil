# -*- coding: utf-8 -*-
"""Búsqueda informada para el rastreo de una incidencia de pago móvil.

Caso hipotético (continuación del informe de búsqueda no informada): una
transacción de pago móvil vence por tiempo de espera en la app de una entidad
financiera. El monitoreo indica que el error de origen está en el conector al
mainframe (CX); para confirmar la causa raíz, el analista debe reconstruir la
traza de la transacción desde la vista (FP) hasta CX, correlacionando la
evidencia entre componentes adyacentes. El costo de cada arco es el tiempo
(en minutos, supuesto de modelado) que toma esa correlación.

Algoritmos: anchura (BFS), costo uniforme (UCS), voraz primero el mejor
(Greedy) y A* con dos heurísticas:
  - h_kappa: costo mínimo por capa multiplicado por las capas que faltan;
  - h_capas: distancia más corta en el grafo abstracto de capas.

Convenciones (las mismas de la unidad):
  - los sucesores se generan en orden alfabético;
  - BFS prueba la meta al GENERAR; UCS, Greedy y A* la prueban al EXTRAER;
  - empates en la cola de prioridad: menor h y luego orden alfabético.

Ejecutar:  python busqueda_informada.py   -> imprime resultados y escribe
resultados_caso.json
"""
import heapq
import json
import statistics
import time
from collections import deque

INF = float("inf")


# --------------------------------------------------------------------------
# 1. Definición formal del problema
# --------------------------------------------------------------------------
class Problema:
    """Grafo dirigido con costos, capa de cada nodo, estado inicial y meta."""

    def __init__(self, arcos, capa, inicial, meta):
        self.arcos = arcos              # {u: {v: costo}}
        self.capa = capa                # {nodo: número de capa}
        self.inicial = inicial
        self.meta = meta
        # Sucesores ordenados una sola vez (orden alfabético de la convención)
        self.suc = {u: sorted(vs.items()) for u, vs in arcos.items()}

    def sucesores(self, n):
        return self.suc.get(n, [])

    def nodos(self):
        return list(self.capa)

    def lista_arcos(self):
        return [(u, v, c) for u, vs in self.arcos.items() for v, c in vs.items()]


# Componentes del caso (los nombres son genéricos, no los de ningún sistema real)
NOMBRES = {
    "FP": "Formulario de pago móvil (vista)",
    "CP": "Controlador del pago",
    "SC": "Servicio de integración de cuentas",
    "SP": "Servicio de integración de pagos",
    "SS": "Servicio de integración de seguridad",
    "AC": "API de cuentas",
    "AP": "API de pagos",
    "AS": "API de seguridad",
    "MC": "Microservicio de caché de saldos",
    "MI": "Microservicio interbancario",
    "MN": "Microservicio de notificaciones",
    "MT": "Microservicio de clave dinámica",
    "MV": "Microservicio de validación biométrica",
    "PB": "Proveedor biométrico externo",
    "CX": "Conector al mainframe",
    "KC": "Programa del core de cuentas",
    "KP": "Programa del core de pagos",
}

# Arcos dirigidos (sentido de la llamada) con el costo de correlación en minutos
ARCOS = {
    "FP": {"CP": 5},
    "CP": {"SC": 10, "SP": 10, "SS": 10},
    "SC": {"AC": 20},
    "SP": {"AP": 15},
    "SS": {"AS": 15},
    "AC": {"CX": 45, "MC": 10},
    "AP": {"CX": 20, "MI": 25, "MN": 15},
    "AS": {"MT": 10, "MV": 15},
    "MV": {"PB": 30},
    "MT": {"CX": 30},
    "CX": {"KC": 60, "KP": 60},
}

# Capa arquitectónica de cada componente: 0 vista, 1 controlador, 2 integración,
# 3 capa media, 4 microservicios, 5 conector y externos, 6 core
CAPA = {"FP": 0, "CP": 1, "SC": 2, "SP": 2, "SS": 2, "AC": 3, "AP": 3, "AS": 3,
        "MC": 4, "MI": 4, "MN": 4, "MT": 4, "MV": 4, "CX": 5, "PB": 5,
        "KC": 6, "KP": 6}

CASO = Problema(ARCOS, CAPA, "FP", "CX")


# --------------------------------------------------------------------------
# 2. Heurísticas
# --------------------------------------------------------------------------
def h_cero(prob):
    """h(n) = 0: con ella A* se comporta como costo uniforme."""
    return {n: 0 for n in prob.nodos()}


def h_kappa(prob):
    """h1(n) = kappa * max(0, L(meta) - L(n)).

    kappa es el menor costo por capa descendida entre todos los arcos que bajan
    de capa: kappa = min c(u,v) / (L(v) - L(u)). Los arcos laterales (misma capa)
    no entran en el mínimo.
    """
    L, Lm = prob.capa, prob.capa[prob.meta]
    kappa = min(c / (L[v] - L[u]) for u, v, c in prob.lista_arcos() if L[v] > L[u])
    return {n: kappa * max(0, Lm - L[n]) for n in prob.nodos()}


def grafo_abstracto(prob):
    """Arcos entre capas con el costo mínimo de los arcos reales que representan."""
    L, abst = prob.capa, {}
    for u, v, c in prob.lista_arcos():
        a, b = L[u], L[v]
        if a != b:                      # los arcos laterales serían bucles
            abst[(a, b)] = min(c, abst.get((a, b), INF))
    return abst


def h_capas(prob):
    """h2(n) = distancia más corta, en el grafo abstracto de capas, de L(n) a L(meta).

    Se calcula con Dijkstra hacia atrás desde la capa de la meta. Una capa sin
    camino abstracto a la meta recibe h = infinito (ningún nodo suyo puede
    llegar a la meta, así que puede podarse).
    """
    abst = grafo_abstracto(prob)
    inversa = {}
    for (a, b), c in abst.items():
        inversa.setdefault(b, []).append((a, c))
    dist, cola = {prob.capa[prob.meta]: 0}, [(0, prob.capa[prob.meta])]
    while cola:
        d, b = heapq.heappop(cola)
        if d > dist[b]:
            continue
        for a, c in inversa.get(b, []):
            if d + c < dist.get(a, INF):
                dist[a] = d + c
                heapq.heappush(cola, (d + c, a))
    return {n: dist.get(prob.capa[n], INF) for n in prob.nodos()}


def costo_real_a_meta(prob):
    """h*(n): costo óptimo real de n a la meta (Dijkstra sobre el grafo invertido)."""
    inversa = {}
    for u, v, c in prob.lista_arcos():
        inversa.setdefault(v, []).append((u, c))
    dist, cola = {prob.meta: 0}, [(0, prob.meta)]
    while cola:
        d, v = heapq.heappop(cola)
        if d > dist[v]:
            continue
        for u, c in inversa.get(v, []):
            if d + c < dist.get(u, INF):
                dist[u] = d + c
                heapq.heappush(cola, (d + c, u))
    return {n: dist.get(n, INF) for n in prob.nodos()}


def validar(prob, h):
    """Comprueba numéricamente las propiedades demostradas en el informe."""
    hstar = costo_real_a_meta(prob)
    arcos = prob.lista_arcos()
    return {
        "h_meta_cero": h[prob.meta] == 0,
        "admisible": all(h[n] <= hstar[n] for n in prob.nodos()),
        "violaciones_admisibilidad": sum(h[n] > hstar[n] for n in prob.nodos()),
        "consistente": all(h[u] <= c + h[v] for u, v, c in arcos),
        "violaciones_consistencia": sum(h[u] > c + h[v] for u, v, c in arcos),
        "arcos_revisados": len(arcos),
    }


# --------------------------------------------------------------------------
# 3. Algoritmos de búsqueda
# --------------------------------------------------------------------------
def reconstruir(padre, n):
    ruta = [n]
    while padre[ruta[-1]] is not None:
        ruta.append(padre[ruta[-1]])
    return ruta[::-1]


def anchura(prob, traza=False):
    """BFS de grafos: frontera FIFO (deque) y meta al generar."""
    s, meta = prob.inicial, prob.meta
    if s == meta:
        return dict(ruta=[s], costo=0, expandidos=0, generados=1, fmax=1, traza=[])
    frontera, padre = deque([s]), {s: None}   # padre también marca "ya alcanzado"
    expandidos, generados, fmax, pasos = 0, 1, 1, []
    while frontera:
        n = frontera.popleft()
        expandidos += 1
        for h, _ in prob.sucesores(n):
            if h in padre:                     # ya está en frontera o explorados
                continue
            padre[h] = n
            generados += 1
            if h == meta:
                ruta = reconstruir(padre, h)
                if traza:
                    pasos.append(dict(nodo=n, frontera=list(frontera)))
                return dict(ruta=ruta, costo=costo_ruta(prob, ruta), expandidos=expandidos,
                            generados=generados, fmax=fmax, traza=pasos)
            frontera.append(h)
        fmax = max(fmax, len(frontera))
        if traza:
            pasos.append(dict(nodo=n, frontera=list(frontera)))
    return None


def mejor_primero(prob, h, modo, traza=False):
    """Búsqueda primero el mejor sobre grafos, con cola de prioridad (heapq).

    modo = "astar"  -> prioridad f(n) = g(n) + h(n)
    modo = "greedy" -> prioridad f(n) = h(n)
    Con h = 0 y modo "astar" se obtiene la búsqueda de costo uniforme.
    La meta se prueba al extraer el nodo. Las entradas viejas de la cola se
    descartan al salir (eliminación perezosa) en lugar de buscarlas y borrarlas.
    Los nodos con h = infinito no se insertan: no pueden llegar a la meta.
    """
    s, meta = prob.inicial, prob.meta

    def prioridad(g, n):
        return (g + h[n] if modo == "astar" else h[n], h[n], n)  # desempate: menor h, nombre

    g, padre = {s: 0}, {s: None}
    cola = [(prioridad(0, s), s)]
    cerrados = set()
    expandidos, generados, fmax, pasos = 0, 1, 1, []
    while cola:
        (f, _, _), n = heapq.heappop(cola)
        if n in cerrados:                      # entrada vieja: ya se expandió con menor g
            continue
        expandidos += 1
        if n == meta:
            ruta = reconstruir(padre, n)
            if traza:
                pasos.append(dict(nodo=n, g=g[n], h=h[n], f=f, suc=[], meta=True,
                                  frontera=vista_frontera(cola, cerrados, g, h)))
            return dict(ruta=ruta, costo=g[n], expandidos=expandidos,
                        generados=generados, fmax=fmax, traza=pasos)
        cerrados.add(n)
        nuevos = []
        for hijo, c in prob.sucesores(n):
            if hijo in cerrados or h[hijo] == INF:
                continue
            ng = g[n] + c
            if ng < g.get(hijo, INF):
                if hijo not in g:
                    generados += 1
                g[hijo], padre[hijo] = ng, n
                heapq.heappush(cola, (prioridad(ng, hijo), hijo))
                nuevos.append(hijo)
        fmax = max(fmax, len(cola))
        if traza:
            pasos.append(dict(nodo=n, g=g[n], h=h[n], f=f, suc=nuevos,
                              frontera=vista_frontera(cola, cerrados, g, h)))
    return None


def vista_frontera(cola, cerrados, g, h):
    """Frontera vigente (sin entradas viejas), ordenada como la cola, para las trazas."""
    vivos = {}
    for clave, n in cola:
        if n not in cerrados and (n not in vivos or clave < vivos[n]):
            vivos[n] = clave                   # la clave menor es la del g vigente
    return [f"{n}({g[n]}+{h[n]:g})" for n, _ in sorted(vivos.items(), key=lambda x: x[1])]


def costo_ruta(prob, ruta):
    return sum(prob.arcos[a][b] for a, b in zip(ruta, ruta[1:]))


def factor_ramificacion_efectivo(expandidos, d, tol=1e-9):
    """b* tal que N + 1 = 1 + b* + b*^2 + ... + b*^d (Russell y Norvig), por bisección."""
    if d <= 0:
        return None
    objetivo = expandidos + 1
    lo, hi = 1.0, float(max(2, expandidos))
    while hi - lo > tol:
        mid = (lo + hi) / 2
        total = sum(mid ** i for i in range(d + 1))
        lo, hi = (mid, hi) if total < objetivo else (lo, mid)
    return (lo + hi) / 2


def cronometrar(funcion, repeticiones):
    """Mediana del tiempo de ejecución en microsegundos."""
    tiempos = []
    for _ in range(repeticiones):
        t0 = time.perf_counter_ns()
        funcion()
        tiempos.append((time.perf_counter_ns() - t0) / 1000)
    return statistics.median(tiempos)


def ejecutar_todos(prob, repeticiones=1, traza=False):
    """Corre las cinco estrategias y registra nodos, tiempo y optimalidad."""
    H = {"cero": h_cero(prob), "kappa": h_kappa(prob), "capas": h_capas(prob)}
    estrategias = {
        "BFS": lambda t=False: anchura(prob, t),
        "UCS": lambda t=False: mejor_primero(prob, H["cero"], "astar", t),
        "Greedy": lambda t=False: mejor_primero(prob, H["capas"], "greedy", t),
        "A*_h1": lambda t=False: mejor_primero(prob, H["kappa"], "astar", t),
        "A*_h2": lambda t=False: mejor_primero(prob, H["capas"], "astar", t),
    }
    res = {k: f(traza) for k, f in estrategias.items()}
    optimo = res["UCS"]["costo"]
    for k, r in res.items():
        r["optimo"] = r["costo"] == optimo
        r["tiempo_us"] = cronometrar(estrategias[k], repeticiones)
        r["prof"] = len(r["ruta"]) - 1
        r["bstar"] = factor_ramificacion_efectivo(r["expandidos"], r["prof"])
    return res, H


if __name__ == "__main__":
    res, H = ejecutar_todos(CASO, repeticiones=2000, traza=True)
    hstar = costo_real_a_meta(CASO)
    print("Heurísticas por nodo (h1, h2, h*):")
    for n in CASO.nodos():
        print(f"  {n}: capa {CAPA[n]}  h1={H['kappa'][n]:g}  h2={H['capas'][n]:g}  h*={hstar[n]:g}")
    print("Grafo abstracto:", grafo_abstracto(CASO))
    for nombre in ("kappa", "capas"):
        print(f"Validación {nombre}:", validar(CASO, H[nombre]))
    for k, r in res.items():
        print(f"\n== {k}: ruta={'-'.join(r['ruta'])} costo={r['costo']} óptimo={r['optimo']} "
              f"expandidos={r['expandidos']} generados={r['generados']} fmax={r['fmax']} "
              f"tiempo={r['tiempo_us']:.1f} µs b*={r['bstar']:.3f}")
        for p in r["traza"]:
            print("  ", p)
    salida = dict(resultados=res, h={k: {n: (None if v == INF else v) for n, v in d.items()}
                                     for k, d in H.items()},
                  hstar={n: (None if v == INF else v) for n, v in hstar.items()},
                  abstracto={f"{a}->{b}": c for (a, b), c in grafo_abstracto(CASO).items()},
                  validacion={k: validar(CASO, H[k]) for k in ("kappa", "capas")},
                  arcos=ARCOS, capa=CAPA, nombres=NOMBRES)
    with open("resultados_caso.json", "w", encoding="utf-8") as fh:
        json.dump(salida, fh, ensure_ascii=False, indent=1)

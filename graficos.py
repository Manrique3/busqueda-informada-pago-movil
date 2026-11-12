# -*- coding: utf-8 -*-
"""Genera los gráficos del informe (python graficos.py).

Lee resultados_caso.json y resultados_escala.json. Todo va en escala de grises
(tinta negra, como pide el manual); las series se distinguen por marcador y
tipo de línea, además de la leyenda.
"""
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"
C = json.load(open("resultados_caso.json", encoding="utf-8"))
E = json.load(open("resultados_escala.json", encoding="utf-8"))
ARCOS, H2 = C["arcos"], C["h"]["capas"]


def nodo(ax, x, y, txt, estilo="normal", r=0.40, fs=12.5):
    fc, lw, ls, col = "white", 1.0, "-", "black"
    if estilo == "meta":
        fc, col, lw = "black", "white", 1.5
    elif estilo == "expandido":
        fc = "#c8c8c8"
    elif estilo == "frontera":
        ls = "--"
    elif estilo == "inicial":
        lw = 2.4
    ax.add_patch(plt.Circle((x, y), r, fc=fc, ec="black", lw=lw, ls=ls, zorder=3))
    ax.text(x, y, txt, ha="center", va="center", fontsize=fs, color=col, zorder=4,
            fontweight="bold")


def flecha(ax, p, q, txt=None, grueso=False, r=0.40, fs=12, dx=0.0, frac=0.5):
    (x1, y1), (x2, y2) = p, q
    d = math.hypot(x2 - x1, y2 - y1)
    ux, uy = (x2 - x1) / d, (y2 - y1) / d
    ax.annotate("", xy=(x2 - ux * r, y2 - uy * r), xytext=(x1 + ux * r, y1 + uy * r),
                arrowprops=dict(arrowstyle="-|>", lw=2.4 if grueso else 0.9, color="black",
                                mutation_scale=8 if grueso else 10), zorder=2)
    if txt is not None:
        ax.text(x1 + (x2 - x1) * frac + dx, y1 + (y2 - y1) * frac, txt, fontsize=fs, ha="center", va="center",
                zorder=5, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))


# ---------- Gráfico 1: espacio de estados con h2 ----------
POS = {"FP": (4, 6), "CP": (4, 5),
       "SC": (1.5, 4), "SP": (4, 4), "SS": (6.5, 4),
       "AC": (1.5, 3), "AP": (4, 3), "AS": (6.5, 3),
       "MC": (0.2, 2), "MN": (3.6, 2), "MI": (5.0, 2), "MT": (6.5, 2), "MV": (8.0, 2),
       "CX": (1.6, 0.9), "PB": (8.0, 0.9),
       "KC": (0.9, -0.2), "KP": (2.3, -0.2)}
YS = 1.5
POS = {k: (x, y * YS) for k, (x, y) in POS.items()}
# Posición de la etiqueta h de cada nodo: a la izquierda donde a la derecha llega una flecha
IZQ = {"SC", "CX", "PB", "KC"}
ABAJO = {"MC", "MN", "MI", "MT"}
# Costo del arco MT→CX más cerca de CX, para no chocar con las etiquetas de MN y MI
FRAC = {("MT", "CX"): 0.75}
# Costo de AS→MT a la izquierda de la flecha, para no chocar con el de AS→MV
DX = {("AS", "MT"): -0.38}
fig, ax = plt.subplots(figsize=(6.0, 5.6))
for a, hs in ARCOS.items():
    for b, c in hs.items():
        vertical = abs(POS[a][0] - POS[b][0]) < 0.01
        dx = DX.get((a, b), 0.38 if vertical else 0)
        flecha(ax, POS[a], POS[b], str(c), dx=dx, frac=FRAC.get((a, b), 0.5))
for n, (x, y) in POS.items():
    nodo(ax, x, y, n, "meta" if n == "CX" else ("inicial" if n == "FP" else "normal"))
    h = H2[n]
    txt = "h=∞" if h is None else f"h={h:g}"
    if n in ABAJO:
        xt, yt, ha = x, y - 0.62, "center"
    elif n in IZQ:
        xt, yt, ha = x - 0.46, y, "right"
    else:
        xt, yt, ha = x + 0.46, y, "left"
    ax.text(xt, yt, txt, fontsize=11, ha=ha, va="center", style="italic", zorder=6,
            bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none"))
capas = [(6, "0 Vista"), (5, "1 Controlador"), (4, "2 Integración"), (3, "3 Capa media"),
         (2, "4 Microservicios"), (0.9, "5 Conector y\n   externos"), (-0.2, "6 Core")]
for y, t in capas:
    ax.text(9.6, y * YS, t, fontsize=11.5, va="center", ha="left")
ax.set_xlim(-0.4, 12.3); ax.set_ylim(-0.75 * YS, 6.45 * YS); ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout()
fig.savefig("grafico1_espacio_estados.png", dpi=220, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)


# ---------- Gráfico 2: árboles de búsqueda (UCS, Greedy, A* con h2) ----------
def arbol_desde_traza(res):
    """Reconstruye el árbol (padre de cada nodo generado) a partir de la traza."""
    padres = {res["traza"][0]["nodo"]: None}
    for p in res["traza"]:
        for h in p["suc"]:
            padres[h] = p["nodo"]
    # Los nodos regenerados con mejor g cambian de padre: se respeta la ruta final
    ruta = res["ruta"]
    for a, b in zip(ruta, ruta[1:]):
        padres[b] = a
    return padres


def arbol(ax, res, titulo, etiqueta):
    padres = arbol_desde_traza(res)
    hijos = {}
    for h, p in padres.items():
        if p is not None:
            hijos.setdefault(p, []).append(h)
    for k in hijos:
        hijos[k].sort()
    pos, cont = {}, [0]

    def lay(n, d):
        if n not in hijos:
            pos[n] = (cont[0] * 2.0, -d * 1.25); cont[0] += 1
        else:
            for h in hijos[n]:
                lay(h, d + 1)
            xs = [pos[h][0] for h in hijos[n]]
            pos[n] = ((min(xs) + max(xs)) / 2, -d * 1.25)
    lay("FP", 0)
    expandidos = {p["nodo"] for p in res["traza"]}
    ruta = res["ruta"]
    en_ruta = set(zip(ruta, ruta[1:]))
    for p, hs in hijos.items():
        for h in hs:
            flecha(ax, pos[p], pos[h], None, grueso=(p, h) in en_ruta)
    for n, (x, y) in pos.items():
        est = "meta" if n == "CX" else ("expandido" if n in expandidos else "frontera")
        nodo(ax, x, y, n, est, fs=11)
        ax.text(x + 0.44, y - 0.24, etiqueta(n), fontsize=10, ha="left", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.05", fc="white", ec="none"))
    ax.set_title(titulo, fontsize=12, fontweight="bold")
    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 0.6, max(xs) + 1.2); ax.set_ylim(min(ys) - 0.6, 0.6)
    ax.set_aspect("equal"); ax.axis("off")


R = C["resultados"]
fig = plt.figure(figsize=(7.2, 6.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
axs = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]


def etiqueta_g(res):
    """g de cada nodo generado, leído de las fronteras de la traza (formato N(g+h))."""
    g = {}
    for p in res["traza"]:
        g[p["nodo"]] = p["g"]
        for item in p["frontera"]:
            n, resto = item.split("(")
            g.setdefault(n, int(resto.split("+")[0]))
    return g


gu, gg, ga = etiqueta_g(R["UCS"]), etiqueta_g(R["Greedy"]), etiqueta_g(R["A*_h2"])
arbol(axs[0], R["UCS"], "(a) Costo uniforme: 13 expandidos", lambda n: f"g={gu.get(n, '')}")
arbol(axs[1], R["Greedy"], "(b) Voraz (h2): 5", lambda n: f"h={H2[n]:g}" if H2[n] is not None else "")
arbol(axs[2], R["A*_h2"], "(c) A* (h2): 6",
      lambda n: f"f={ga.get(n, 0) + H2[n]:g}" if H2[n] is not None else "")
fig.tight_layout()
fig.savefig("grafico2_arboles.png", dpi=220, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)


# ---------- Gráficos 3 y 4: experimentos ----------
ESTILO = {
    "BFS": dict(marker="s", ls=":", color="#7a7a7a", label="Anchura"),
    "UCS": dict(marker="o", ls="-", color="black", label="Costo uniforme"),
    "Greedy": dict(marker="v", ls="-.", color="#7a7a7a", label="Voraz (h2)"),
    "A*_h1": dict(marker="^", ls="--", color="#4a4a4a", label="A* (h1)"),
    "A*_h2": dict(marker="D", ls="-", color="#4a4a4a", label="A* (h2)", mfc="white"),
}


def ejes(ax, titulo, xlabel, ylabel):
    ax.set_title(titulo, fontsize=9.5, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=8.5); ax.set_ylabel(ylabel, fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.grid(True, color="#e3e3e3", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


esc = E["escala"]
ns = [f["n"] for f in esc]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.3))
for k, st in ESTILO.items():
    a1.plot(ns, [f[k]["expandidos_media"] for f in esc], lw=1.6, ms=5, **st)
    a2.plot(ns, [f[k]["tiempo_ms_media"] for f in esc], lw=1.6, ms=5, **st)
for a in (a1, a2):
    a.set_xscale("log")
ejes(a1, "(a) Nodos expandidos", "Componentes de la arquitectura (escala log.)", "Media de nodos expandidos")
ejes(a2, "(b) Tiempo de ejecución", "Componentes de la arquitectura (escala log.)", "Media de tiempo (ms)")
a1.legend(fontsize=7.5, frameon=False)
fig.tight_layout()
fig.savefig("grafico3_escala.png", dpi=220, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)

sen = E["sensibilidad"]
ds = [f["dispersion"] for f in sen]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
for k in ("A*_h1", "A*_h2"):
    a1.plot(ds, [100 * f[k]["expandidos_media"] / f["UCS"]["expandidos_media"] for f in sen],
            lw=1.6, ms=5, **ESTILO[k])
a1.axhline(100, color="black", lw=0.8)
a1.text(ds[-1], 101.5, "Costo uniforme = 100", fontsize=7.5, ha="right")
a1.set_ylim(60, 106)
for k in ("BFS", "Greedy"):
    a2.plot(ds, [f[k]["pct_optimo"] for f in sen], lw=1.6, ms=5, **ESTILO[k])
a2.set_ylim(0, 105)
ejes(a1, "(a) Expandidos respecto a costo uniforme", "Dispersión de costos (r en U(1; r))", "% de los expandidos por UCS")
ejes(a2, "(b) Instancias resueltas con la ruta óptima", "Dispersión de costos (r en U(1; r))", "% de instancias")
a1.legend(fontsize=7.5, frameon=False, loc="lower right")
a2.legend(fontsize=7.5, frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig("grafico4_sensibilidad.png", dpi=220, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print("ok")

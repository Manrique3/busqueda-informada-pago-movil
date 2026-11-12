# -*- coding: utf-8 -*-
"""Experimentos de escalabilidad y sensibilidad sobre arquitecturas por capas.

El caso de 17 componentes es demasiado pequeño para medir tiempos con sentido,
así que se generan arquitecturas sintéticas con la misma estructura (7 capas,
del frontend al core) y tamaños crecientes. En cada instancia se corren las
cinco estrategias de busqueda_informada.py y se registran nodos expandidos,
tiempo de ejecución y optimalidad. También se valida, instancia por instancia,
que h1 y h2 sean admisibles y consistentes.

Experimento 1 (escala): tamaños de 100 a 20.000 componentes, dispersión 3.
Experimento 2 (sensibilidad): 5.000 componentes y dispersión de 1 a 5, para ver
cómo la variedad de costos dentro de una misma transición entre capas afecta a h2.

Supuestos de generación (declarados en el informe):
  - capa 0: una sola vista (estado inicial); capas 1 a 6: el mismo número de nodos;
  - cada nodo de las capas 0 a 5 llama a entre 2 y 6 nodos de la capa siguiente;
  - con probabilidad 0,15 añade un arco que salta una capa y con 0,20 uno lateral;
  - costo de un arco = costo base de esa transición x U(1; dispersión), redondeado
    (los saltos suman los costos base; los laterales cuestan entre 5 y 30 min);
  - la meta es un nodo de la capa 5 (conectores) alcanzable desde la vista.

Ejecutar:  python experimento_escala.py  -> escribe resultados_escala.json
"""
import json
import random
import statistics

from busqueda_informada import Problema, ejecutar_todos, h_capas, h_kappa, validar

# Costo base (min) de bajar de la capa i a la i+1, tomado del orden de magnitud del caso
BASE = [5, 10, 15, 10, 25, 60]
GRADO = (2, 6)          # llamadas de cada nodo a la capa siguiente (mínimo, máximo)
TAMANOS = [100, 500, 1000, 5000, 10000, 20000]
DISPERSIONES = [1.0, 1.5, 2.0, 3.0, 5.0]
N_SENSIBILIDAD = 5000
INSTANCIAS = 30
REPETICIONES = 3
SEMILLA = 2026


def generar(n_total, rng, dispersion=3.0):
    ancho = max(2, (n_total - 1) // 6)
    capas = [["V0"]] + [[f"N{c}_{i}" for i in range(ancho)] for c in range(1, 7)]
    capa = {n: c for c, nodos in enumerate(capas) for n in nodos}
    arcos = {}

    def arco(u, v, c):
        if u != v:
            arcos.setdefault(u, {})[v] = min(c, arcos.get(u, {}).get(v, c))

    for c in range(6):
        for u in capas[c]:
            for v in rng.sample(capas[c + 1], rng.randint(GRADO[0], min(GRADO[1], len(capas[c + 1])))):
                arco(u, v, round(BASE[c] * rng.uniform(1, dispersion)))
            if c < 5 and rng.random() < 0.15:
                arco(u, rng.choice(capas[c + 2]),
                     round((BASE[c] + BASE[c + 1]) * rng.uniform(1, dispersion)))
            if c > 0 and rng.random() < 0.20:
                arco(u, rng.choice(capas[c]), rng.randint(5, 30))
    # Meta: un conector (capa 5) alcanzable desde la vista
    alcanzables = alcanzables_desde(Problema(arcos, capa, "V0", capas[5][0]))
    meta = rng.choice([n for n in capas[5] if n in alcanzables])
    return Problema(arcos, capa, "V0", meta)


def alcanzables_desde(prob):
    vistos, pila = {prob.inicial}, [prob.inicial]
    while pila:
        for v, _ in prob.sucesores(pila.pop()):
            if v not in vistos:
                vistos.add(v)
                pila.append(v)
    return vistos


def correr_serie(etiqueta, valores, fabricar, rng, validaciones):
    """Corre INSTANCIAS problemas por cada valor y promedia las métricas."""
    resumen = []
    for x in valores:
        acumulado, alcanzables = {}, []
        for _ in range(INSTANCIAS):
            prob = fabricar(x, rng)
            alcanzables.append(len(alcanzables_desde(prob)))
            for nombre, h in (("h1", h_kappa(prob)), ("h2", h_capas(prob))):
                v = validar(prob, h)
                validaciones[f"violaciones_{nombre}"] += (v["violaciones_admisibilidad"]
                                                         + v["violaciones_consistencia"])
                validaciones["arcos_revisados"] += v["arcos_revisados"]
            validaciones["instancias"] += 1
            res, _ = ejecutar_todos(prob, repeticiones=REPETICIONES)
            optimo = res["UCS"]["costo"]
            for k, r in res.items():
                a = acumulado.setdefault(k, {"exp": [], "gen": [], "t": [], "opt": [], "ratio": [],
                                             "bstar": []})
                a["exp"].append(r["expandidos"])
                a["gen"].append(r["generados"])
                a["t"].append(r["tiempo_us"] / 1000)
                a["opt"].append(r["optimo"])
                a["ratio"].append(r["costo"] / optimo)
                a["bstar"].append(r["bstar"])
        fila = {etiqueta: x, "alcanzables_media": statistics.mean(alcanzables)}
        for k, a in acumulado.items():
            fila[k] = {
                "expandidos_media": statistics.mean(a["exp"]),
                "generados_media": statistics.mean(a["gen"]),
                "tiempo_ms_media": statistics.mean(a["t"]),
                "pct_optimo": 100 * sum(a["opt"]) / len(a["opt"]),
                "costo_relativo_media": statistics.mean(a["ratio"]),
                "costo_relativo_max": max(a["ratio"]),
                "bstar_media": statistics.mean(a["bstar"]),
            }
        resumen.append(fila)
        print(etiqueta, x, round(fila["alcanzables_media"]),
              {k: (round(v["expandidos_media"], 1), round(v["tiempo_ms_media"], 3),
                   round(v["pct_optimo"]), round(v["costo_relativo_media"], 3))
               for k, v in fila.items() if isinstance(v, dict)})
    return resumen


def main():
    rng = random.Random(SEMILLA)
    validaciones = {"instancias": 0, "arcos_revisados": 0, "violaciones_h1": 0, "violaciones_h2": 0}
    escala = correr_serie("n", TAMANOS, lambda n, r: generar(n, r), rng, validaciones)
    sensibilidad = correr_serie("dispersion", DISPERSIONES,
                                lambda d, r: generar(N_SENSIBILIDAD, r, d), rng, validaciones)
    print("Validación:", validaciones)
    with open("resultados_escala.json", "w", encoding="utf-8") as fh:
        json.dump({"escala": escala, "sensibilidad": sensibilidad, "validacion": validaciones,
                   "parametros": {"tamanos": TAMANOS, "dispersiones": DISPERSIONES,
                                  "n_sensibilidad": N_SENSIBILIDAD, "instancias": INSTANCIAS,
                                  "repeticiones": REPETICIONES, "semilla": SEMILLA,
                                  "base": BASE, "grado": GRADO}},
                  fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

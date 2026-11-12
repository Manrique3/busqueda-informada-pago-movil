# Búsqueda informada con A* para rastrear una incidencia de pago móvil

Código de la simulación del informe técnico *Búsqueda informada con A\* para rastrear una incidencia de pago móvil* (Inteligencia Artificial, MIAP-I13, Maestría en Ingeniería en Inteligencia Artificial, Universidad Bicentenaria de Aragua).

El caso es **hipotético**: una transacción de pago móvil vence por tiempo de espera y hay que reconstruir la traza desde la vista de la app (FP) hasta el conector al mainframe (CX). Los nombres de los componentes son genéricos y los costos en minutos son supuestos de modelado.

## Contenido

| Archivo | Qué hace |
|---|---|
| `busqueda_informada.py` | Define el problema (grafo, capas, inicio y meta), las heurísticas h1 (costo mínimo por capa) y h2 (distancia en el grafo abstracto de capas), los algoritmos (anchura, costo uniforme, voraz y A*) y la validación numérica de admisibilidad y consistencia. Corre el caso de 17 componentes. |
| `experimento_escala.py` | Genera arquitecturas sintéticas de 100 a 20.000 componentes (escala) y de 5.000 componentes con dispersión de costos de 1 a 5 (sensibilidad), 30 instancias por valor, con semilla fija. |
| `graficos.py` | Produce los cuatro gráficos del informe a partir de los resultados. |
| `resultados_caso.json`, `resultados_escala.json` | Resultados usados en el informe. |

## Cómo reproducir

Requiere Python 3.12. Los algoritmos no usan librerías externas; los gráficos usan matplotlib.

```
python busqueda_informada.py     # caso: trazas y resultados_caso.json
python experimento_escala.py     # experimentos: resultados_escala.json (unos minutos)
python graficos.py               # grafico1 a grafico4 (.png)
```

Los conteos de nodos son deterministas. Los tiempos dependen del equipo y varían entre ejecuciones (el informe usó un AMD Ryzen 7 5825U con Windows 11).

## Convenciones

- Sucesores en orden alfabético.
- Anchura prueba la meta al generar; costo uniforme, voraz y A* la prueban al extraer.
- Empates en la cola de prioridad: menor h y luego orden alfabético.
- Los nodos con h = ∞ no se insertan en la frontera (no pueden llegar a la meta).

## Autor

Gabriel Manrique, 2026.

# Resumen de entrenamientos

Este documento resume los resultados de entrenamiento recibidos para diferentes elementos del ciclo del kiwi.

El objetivo no es presentar modelos productivos, sino documentar el trabajo experimental realizado, las métricas obtenidas y el estado relativo de cada línea de entrenamiento.

## Entrenamientos documentados

| Dataset / clase | Épocas | Precision final | Recall final | mAP50 final | mAP50-95 final |
|---|---:|---:|---:|---:|---:|
| [Yemas](entrenamiento-yemas.md) | 100 | 61.96% | 43.50% | 46.69% | 18.98% |
| [Brotes](entrenamiento-brotes.md) | 70 | 97.56% | 95.39% | 98.26% | 76.42% |
| [Flor](entrenamiento-flor.md) | 60 | 58.39% | 37.88% | 40.89% | 16.13% |
| [Kiwis](entrenamiento-kiwis.md) | 50 | 90.94% | 83.13% | 91.84% | 68.09% |

## Lectura general

Los resultados muestran comportamientos diferentes según la clase trabajada:

- **Brotes**: resultados altos, con buena precisión, recall y mAP. Es el entrenamiento más sólido de los recibidos.
- **Kiwis**: resultados buenos para detección de frutos. Puede servir como base para pruebas de conteo y validación visual.
- **Yemas**: resultados intermedios/bajos. Requiere refuerzo de dataset y revisión de etiquetas.
- **Flor**: resultados bajos/intermedios. Requiere más trabajo de dataset, variabilidad de imágenes y validación.

## Archivos disponibles

- [Entrenamiento de yemas](entrenamiento-yemas.md)
- [Entrenamiento de brotes](entrenamiento-brotes.md)
- [Entrenamiento de flor](entrenamiento-flor.md)
- [Entrenamiento de kiwis](entrenamiento-kiwis.md)

## Consideraciones

No se incluyen datasets completos ni modelos entrenados pesados.

Las métricas deben tomarse como resultados de una etapa de experimentación. Para una validación más firme habría que complementar estos reportes con:

- cantidad exacta de imágenes por dataset;
- cantidad de anotaciones;
- distribución train/val/test;
- ejemplos de falsos positivos;
- ejemplos de falsos negativos;
- validación con imágenes nuevas de campo.

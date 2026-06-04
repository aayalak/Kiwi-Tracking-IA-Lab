# Procesamiento de imágenes

## Objetivo

Procesar imágenes capturadas en campo para detectar elementos del cultivo de kiwi mediante modelos de detección de objetos.

## Elementos considerados

- Ramas
- Yemas
- Brotes
- Flores
- Hojas
- Frutos
- Anomalías en frutos
- Anomalías en hojas
- Plagas

## Resultado esperado por imagen

Cada imagen procesada debería permitir registrar:

- cantidad de elementos detectados;
- coordenadas de cada detección;
- clase detectada;
- nivel de confianza;
- imagen original;
- imagen anotada;
- fecha y ubicación de captura;
- recorrido asociado.

## Consideraciones

Los resultados dependen de:

- calidad de las imágenes;
- iluminación;
- distancia al objeto;
- ángulo de captura;
- calidad del dataset;
- modelo utilizado;
- cantidad y calidad de etiquetas.

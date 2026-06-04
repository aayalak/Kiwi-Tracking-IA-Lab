# Arquitectura y etapas

## Enfoque general

El proyecto fue pensado con una arquitectura modular, separando la gestión de datos productivos del servicio de detección por imágenes.

La idea inicial contempla:

- una aplicación de gestión;
- una API de procesamiento de imágenes;
- almacenamiento de capturas;
- base de datos para recorridos, metadata y resultados;
- modelos de IA entrenados para detección;
- visualización y reportes.

## MVP

Para una primera etapa se propuso:

- Python + FastAPI;
- SQLite como base local;
- almacenamiento de imágenes en carpetas;
- procesamiento con modelo YOLO;
- consulta de resultados mediante API o interfaz simple.

## Evolución posible

En etapas posteriores se evaluó:

- PostgreSQL/PostGIS para datos georreferenciados;
- dashboards y mapas;
- almacenamiento en nube;
- integración con datos satelitales;
- mejora continua de modelos con nuevas imágenes.

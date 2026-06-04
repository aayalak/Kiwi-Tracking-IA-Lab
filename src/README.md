# KiwiTrackingIA - Portfolio

Version curada para portfolio profesional de una aplicacion de vision por computadora para seguimiento y analisis de kiwis.

## Objetivo

Mostrar una integracion completa entre FastAPI, YOLO, OpenCV y procesamiento de imagenes para:

- deteccion en tiempo real desde camara;
- procesamiento de videos;
- analisis por imagen para kiwis, flores, brotes y yemas;
- muestreo adaptativo de frames con SSIM y flujo optico;
- visualizacion web con templates HTML.

## Stack

- Python
- FastAPI
- Jinja2
- OpenCV
- PyTorch
- Ultralytics YOLO
- scikit-image
- Bootstrap

## Estructura

```text
app/
  main.py
utils/
  video_sampler.py
templates/
static/
examples/
docs/
```

## Modelos

Los modelos entrenados no se incluyen en este repositorio por peso y buenas practicas de publicacion. El codigo espera modelos YOLO en una carpeta local `modelos/`.

Ver `docs/MODELOS.md` para los nombres esperados y la politica recomendada para distribuirlos.

## Ejecucion local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

La aplicacion queda disponible en:

```text
http://127.0.0.1:8000
```

## Nota de portfolio

Esta carpeta excluye datasets, capturas, recortes, resultados, logs, videos, bases locales y pesos entrenados. El objetivo es mostrar arquitectura y codigo representativo sin publicar datos operativos o archivos pesados.


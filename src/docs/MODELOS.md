# Modelos YOLO

Los pesos entrenados fueron excluidos deliberadamente del portfolio.

## Rutas esperadas por el codigo

El archivo `app/main.py` referencia modelos dentro de una carpeta local `modelos/`:

```text
modelos/YOLOv8M_best_072025.pt
modelos/YOLOv8M_Flor_kiwi_best.pt
modelos/YOLOv8M_brote_flor_best.pt
modelos/YOLOv8M_yemas_best.pt
```

## Motivo de exclusion

- Los archivos `.pt` son pesados para GitHub.
- Los pesos entrenados pueden representar propiedad intelectual del proyecto.
- GitHub no es el lugar ideal para versionar binarios de modelos.

## Recomendacion

Publicar los modelos por alguno de estos canales:

- GitHub Releases;
- Hugging Face Hub;
- almacenamiento S3 compatible;
- Google Drive o storage privado para demos controladas.


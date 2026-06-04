# Ejemplo de request para procesar video

Endpoint:

```text
POST /procesar_video
Content-Type: multipart/form-data
```

Campo esperado:

```text
video_file: archivo MP4
```

Ejemplo con curl:

```bash
curl -X POST \
  -F "video_file=@demo.mp4" \
  http://127.0.0.1:8000/procesar_video
```

El video real no se incluye en el portfolio.


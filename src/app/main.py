from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
from fastapi import UploadFile, File, Request, Form
import random
import time
import os
import json
from fastapi import UploadFile, File
import shutil
import cv2
from fastapi.responses import StreamingResponse
from datetime import datetime
import torch
import threading
import numpy as np
from typing import Optional
from ultralytics import YOLO


# Variables globales para estado de detección
#-- Variables globales --
#Variables para manejar el estado de la detección y los frames - TIEMPO REAL
detector_running = False
frame_global = None
detections_global = None
lock = threading.Lock()
camera_thread = None
etector_thread = None
# Variables para manejar el estado de la detección y los frames - VIDEO
frame_global_video = None
detections_global_video = None
lock_video = threading.Lock()
video_running = False
#-----------------------------------------------

#-------- Ruta del modelo YOLOv8
#-- Modelo de solo Kiwis ---
MODELO_PATH_Yolov8 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modelos', 'YOLOv8M_best_072025.pt'))
#-- Modelo de solo Flores determina si es hembra o macho---
MODELO_PATH_Yolov8_Flor = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modelos', 'YOLOv8M_Flor_kiwi_best.pt'))
#-- Modelo de brotes y flores  ---
MODELO_PATH_Yolov8_Flor_Brotes = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modelos', 'YOLOv8M_brote_flor_best.pt'))
#-- Modelo de yemas  ---
MODELO_PATH_Yolov8_Yemas = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modelos', 'YOLOv8M_yemas_best.pt'))
#--------carpetas para capturas y recortes
CAPTURAS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'capturas'))
RECORTES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'recortes'))
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'BaseDatos', 'capturas.json'))
#--------carpetas para adjuntar archivos de imagenes de kiwi
ANALISIS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'analisis_imagenes'))
ANALISIS_DIR_FLOR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'analisis_imagenes_Flor'))
ANALISIS_DIR_FLOR_BROTE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'analisis_imagenes_Flor_Brote'))
ANALISIS_DIR_YEMAS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'analisis_imagenes_Yemas'))
RECORTES_ANALISIS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'recortes_analisis'))
RECORTES_ANALISIS_DIR_FLOR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'recortes_analisis_flor'))
RECORTES_ANALISIS_DIR_FLOR_BROTE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'recortes_analisis_flor_brote'))
RECORTES_ANALISIS_DIR_YEMAS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'recortes_analisis_yemas'))
RESULTADOS_DETECCIONES_DIR_YEMAS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resultados_detecciones_yemas'))
RESULTADOS_DETECCIONES_DIR_FLOR_BROTE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resultados_detecciones_flor_brote'))
RESULTADOS_DETECCIONES_DIR_FLOR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resultados_detecciones_flor'))
RESULTADOS_DETECCIONES_DIR_KIWI = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resultados_detecciones_kiwi'))

ANALISIS_JSON = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'analisis_resultados.json'))
ANALISIS_JSON_FLOR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'analisis_resultados_flor.json'))
ANALISIS_JSON_FLOR_BROTE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'analisis_resultados_flor_Brote.json'))
ANALISIS_JSON_YEMAS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'analisis_resultados_yemas.json'))
MIN_ANCHO = 60    # mínimo ancho en píxeles para un kiwi real
MIN_ALTO = 60        # mínimo alto
ASPECT_RATIO_MIN = 0.7
ASPECT_RATIO_MAX = 1.5
UMBRAL_CONFIDENCE = 0.35  # umbral de confianza para detección de kiwis

# Crear directorios si no existen
os.makedirs(ANALISIS_DIR, exist_ok=True)
os.makedirs(ANALISIS_DIR_FLOR, exist_ok=True)
os.makedirs(ANALISIS_DIR_FLOR_BROTE, exist_ok=True)
os.makedirs(ANALISIS_DIR_YEMAS, exist_ok=True)
os.makedirs(RECORTES_ANALISIS_DIR, exist_ok=True)
os.makedirs(RECORTES_ANALISIS_DIR_FLOR, exist_ok=True)
os.makedirs(RECORTES_ANALISIS_DIR_FLOR_BROTE, exist_ok=True)
os.makedirs(RECORTES_ANALISIS_DIR_YEMAS, exist_ok=True)
os.makedirs(RESULTADOS_DETECCIONES_DIR_FLOR_BROTE, exist_ok=True)
os.makedirs(RESULTADOS_DETECCIONES_DIR_FLOR, exist_ok=True)
os.makedirs(RESULTADOS_DETECCIONES_DIR_KIWI, exist_ok=True)
os.makedirs(RESULTADOS_DETECCIONES_DIR_YEMAS, exist_ok=True)


# Crear archivo JSON de análisis si no existe
if not os.path.exists(ANALISIS_JSON):
    with open(ANALISIS_JSON, 'w', encoding='utf-8') as f:
        json.dump([], f, indent=2, ensure_ascii=False)
# Crear archivo JSON de análisis de flores si no existe
if not os.path.exists(ANALISIS_JSON_FLOR):
    with open(ANALISIS_JSON_FLOR, 'w', encoding='utf-8') as f:
        json.dump([], f, indent=2, ensure_ascii=False)
# Crear archivo JSON de análisis de flores y brotes si no existe
if not os.path.exists(ANALISIS_JSON_FLOR_BROTE):
    with open(ANALISIS_JSON_FLOR_BROTE, 'w', encoding='utf-8') as f:
        json.dump([], f, indent=2, ensure_ascii=False)
# Crear archivo JSON de análisis de yemas si no existe
if not os.path.exists(ANALISIS_JSON_YEMAS):
    with open(ANALISIS_JSON_YEMAS, 'w', encoding='utf-8') as f:
        json.dump([], f, indent=2, ensure_ascii=False)
#
os.makedirs(CAPTURAS_DIR, exist_ok=True)
os.makedirs(RECORTES_DIR, exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'static'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'templates'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'BaseDatos'), exist_ok=True)


app = FastAPI()
# Montar directorios estáticos
app.mount("/capturas", StaticFiles(directory=CAPTURAS_DIR), name="capturas")
app.mount("/recortes", StaticFiles(directory=RECORTES_DIR), name="recortes")
app.mount("/img", StaticFiles(directory=os.path.join(os.path.dirname(__file__), '..', 'img')), name="img")
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), '..', 'static')), name="static")
# Montar directorios de análisis de imágenes
app.mount("/analisis_imagenes", StaticFiles(directory=ANALISIS_DIR), name="analisis_imagenes")
app.mount("/analisis_imagenes_flor", StaticFiles(directory=ANALISIS_DIR_FLOR), name="analisis_imagenes_flor")
app.mount("/analisis_imagenes_flor_brote", StaticFiles(directory=ANALISIS_DIR_FLOR_BROTE), name="analisis_imagenes_flor_brote")
app.mount("/analisis_imagenes_yemas", StaticFiles(directory=ANALISIS_DIR_YEMAS), name="analisis_imagenes_yemas")
# Montar directorios de recortes de análisis
app.mount("/recortes_analisis", StaticFiles(directory=RECORTES_ANALISIS_DIR), name="recortes_analisis")
app.mount("/recortes_analisis_flor", StaticFiles(directory=RECORTES_ANALISIS_DIR_FLOR), name="recortes_analisis_flor")
app.mount("/recortes_analisis_flor_brote", StaticFiles(directory=RECORTES_ANALISIS_DIR_FLOR_BROTE), name="recortes_analisis_flor_brote")
app.mount("/recortes_analisis_yemas", StaticFiles(directory=RECORTES_ANALISIS_DIR_YEMAS), name="recortes_analisis_yemas")
# Montar directorios de resultados de análisis
app.mount("/resultados_detecciones_flor", StaticFiles(directory=RESULTADOS_DETECCIONES_DIR_FLOR), name="resultados_detecciones_flor")
app.mount("/resultados_detecciones_flor_brote", StaticFiles(directory=RESULTADOS_DETECCIONES_DIR_FLOR_BROTE), name="resultados_detecciones_flor_brote")
app.mount("/resultados_detecciones_kiwi", StaticFiles(directory=RESULTADOS_DETECCIONES_DIR_KIWI), name="resultados_detecciones_kiwi")
app.mount("/resultados_detecciones_yemas", StaticFiles(directory=RESULTADOS_DETECCIONES_DIR_YEMAS), name="resultados_detecciones_yemas")

# Configurar Jinja2 para plantillas HTML
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), '..', 'templates'))

# --- Rutas web ---
# --- Cargar capturas previas ---
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "capturando": detector_running,
            "now": datetime.now
        }
    )

@app.post("/subir_video")
async def subir_video(video_file: UploadFile = File(...)):
    # Guardar archivo temporalmente
    video_path = os.path.join(CAPTURAS_DIR, video_file.filename)
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video_file.file, buffer)

    # Procesar el video con OpenCV y YOLO
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Detectar kiwis en el frame igual que en la webcam...
        # Si detecta, guarda la captura y recortes, y agrega registro con "origen": "video", "nombre_archivo": video_file.filename
        # ... (el mismo código del loop de detección en tiempo real)
        frame_idx += 1
        # Si querés, solo guardar una cada X frames para no inundar la galería

    cap.release()
    # Podés borrar el video temporal si no lo necesitás más
    os.remove(video_path)

    return RedirectResponse("/galeria", status_code=303)


@app.post("/iniciar")
def iniciar_captura():
    global detector_running, camera_thread, detector_thread
    if not detector_running:
        detector_running = True
        model = getattr(generate_frames, "model", None)
        camera_thread = threading.Thread(target=camera_reader, daemon=True)
        detector_thread = threading.Thread(target=detector_loop, args=(model,), daemon=True)
        camera_thread.start()
        detector_thread.start()
    return RedirectResponse("/", status_code=303)

@app.post("/detener")
def detener_captura():
    global detector_running
    detector_running = False
    return RedirectResponse("/", status_code=303)


#--- FUNCIONES DE DETECCIÓN ---
# Esta función se ejecuta en un hilo separado para leer la cámara
#Lector de cámara
def camera_reader():
    global frame_global, detector_running
    cap = cv2.VideoCapture(0)
    while detector_running:
        ret, frame = cap.read()
        if not ret:
            break
        with lock:
            frame_global = frame.copy()
        cv2.waitKey(1)
    cap.release()


#Detector (hilo de detección)
def detector_loop(model):
    global frame_global, detections_global, detector_running
    while detector_running:
        with lock:
            frame = frame_global.copy() if frame_global is not None else None
        if frame is None:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        kiwis_detectados = []
        guardar = False

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                clase = model.names[cls_id]
                aspect_ratio = 0
                if clase == 'kiwi' and conf > 0.4:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    ancho = x2 - x1
                    alto = y2 - y1
                    aspect_ratio = ancho / alto if alto != 0 else 0
                    area = ancho * alto
                    # Filtrar SOLO kiwis reales (tamaño y forma)
                    if (
                        clase == 'kiwi'
                        and conf > 0.5
                        and ancho >= MIN_ANCHO
                        and alto >= MIN_ALTO
                        and ASPECT_RATIO_MIN <= aspect_ratio <= ASPECT_RATIO_MAX
                    ):
                        score_pct = int(conf * 100)
                        nombre_recorte = f"Kiwi_recorte_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{cls_id}.jpg"
                        recorte = frame[y1:y2, x1:x2]
                        ok = cv2.imwrite(os.path.join(RECORTES_DIR, nombre_recorte), recorte)
                        kiwis_detectados.append({
                            "clase": "kiwi",
                            "confianza": conf,
                            "confianza_porcentaje": score_pct,
                            "box": [x1, y1, x2, y2],
                            "ancho": ancho,
                            "alto": alto,
                            "area": area,
                            "archivo_recorte": f"recortes/{nombre_recorte}"
                        })
                        guardar = True

        detections_global = kiwis_detectados  # Para usar en los frames

        if guardar:
            nombre_captura = f"captura_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            archivo_captura = os.path.join(CAPTURAS_DIR, nombre_captura)
            ok2 = cv2.imwrite(archivo_captura, frame)
            registro = {
                "timestamp": datetime.now().isoformat(),
                "archivo_captura": f"capturas/{nombre_captura}",
                "origen": "tiempo real",
                "kiwis_detectados": kiwis_detectados
            }
            try:
                with open(DB_PATH, 'r+', encoding='utf-8') as f:
                    data = json.load(f)
                    data.append(registro)
                    f.seek(0)
                    json.dump(data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print("Error actualizando capturas.json:", e)

        time.sleep(0.05)



# --- Procesamiento de video en tiempo real ---
#
def generate_frames():
    global detector_running, frame_global, detections_global
    print("Iniciando generación de frames...")
    model = getattr(generate_frames, "model", None)
    if model is None:
        model = YOLO(MODELO_PATH_Yolov8)
        if torch.cuda.is_available():
            model = model.cuda()
        generate_frames.model = model
    print("Modelo YOLOv8 cargado, comenzando a generar frames...")
    while True:
        if not detector_running:
            frame = 255 * np.ones((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Esperando comenzar...", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (128,128,128), 2)
            ret2, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            import time; time.sleep(0.1)
            continue

        with lock:
            frame = frame_global.copy() if frame_global is not None else None
        if frame is None:
            continue

        results = model(frame)
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                clase = model.names[cls_id]
                if clase == 'kiwi' and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(
                        frame,
                        f"{clase} {int(conf*100)}%",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                        3
                    )
        ret2, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        import time; time.sleep(0.02)


#------------------------------------------------------

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


from datetime import datetime

@app.get("/galeria", response_class=HTMLResponse)
def galeria(request: Request):
    with open(DB_PATH, "r", encoding="utf-8") as f:
        registros = json.load(f)
    registros.sort(key=lambda r: r["timestamp"], reverse=True)
    return templates.TemplateResponse(
        "galeria.html",
        {
            "request": request,
            "registros": registros,
            "now": datetime.now
        }
    )

@app.post("/borrar")
def borrar_captura(archivo_captura: str = Form(...)):
    # Borra la imagen, recortes y entrada del JSON
    with open(DB_PATH, "r+", encoding="utf-8") as f:
        registros = json.load(f)
        registros = [r for r in registros if r["archivo_captura"] != archivo_captura]
        f.seek(0)
        f.truncate()
        json.dump(registros, f, indent=2, ensure_ascii=False)
    # Borra el archivo de imagen
    try:
        os.remove(os.path.join(CAPTURAS_DIR, os.path.basename(archivo_captura)))
    except:
        pass
    # Borra recortes asociados
    # (Esto solo funciona si el campo archivo_recorte tiene el nombre bien formado)
    return RedirectResponse("/galeria", status_code=303)

from fastapi import Request
from fastapi import HTTPException

from fastapi import HTTPException

@app.post("/procesar_video")
async def procesar_video(
    request: Request,
    video_file: UploadFile = File(...)
):
    safe_name = os.path.basename(video_file.filename).replace(" ", "_")
    video_path = os.path.join(CAPTURAS_DIR, safe_name)
    os.makedirs(CAPTURAS_DIR, exist_ok=True)

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video_file.file, buffer)

    # Validar que se lee al menos 1 frame
    cap = cv2.VideoCapture(video_path)
    ok, _ = cap.read()
    cap.release()
    if not ok:
        try: os.remove(video_path)
        except: pass
        raise HTTPException(status_code=400, detail="No se pudo leer el video (codec/archivo inválido).")

    return templates.TemplateResponse(
        "procesando_video.html",
        {"request": request, "video_filename": safe_name, "now": datetime.now}
    )


#-- FUNCIONES DE PROCESAMIENTO DE VIDEO --
def video_reader(video_path):
    global frame_global_video, video_running
    cap = cv2.VideoCapture(video_path)
    while video_running:
        ret, frame = cap.read()
        if not ret:
            break
        with lock_video:
            frame_global_video = frame.copy()
        cv2.waitKey(1)
    cap.release()
    video_running = False

def video_detector_loop(model):
    global frame_global_video, detections_global_video, video_running
    while video_running:
        with lock_video:
            if frame_global_video is not None:
                frame = frame_global_video.copy()
            else:
                continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        detections_global_video = results.pandas().xyxy[0]
        time.sleep(0.1)
#-- FIN FUNCIONES DE PROCESAMIENTO DE VIDEO --

def generate_frames_from_video(filename):
    import os, time, json, cv2, numpy as np
    from datetime import datetime
    import torch

    video_path = os.path.join(CAPTURAS_DIR, filename)

    # Modelo singleton (GPU si hay)
    model = getattr(generate_frames_from_video, "model", None)
    if model is None:
        m = YOLO(MODELO_PATH_Yolov8)
        if torch.cuda.is_available():
            m = m.cuda()
        generate_frames_from_video.model = m
        model = m

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        err = 255 * np.ones((480, 640, 3), dtype=np.uint8)
        cv2.putText(err, "No se pudo abrir el video", (40, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        ok, buf = cv2.imencode(".jpg", err)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        return

    # ==== knobs ====
    EVERY_N = 3                 # correr YOLO 1 de cada N frames
    CONF_TH = max(0.0, float(UMBRAL_CONFIDENCE))  # usa tu global (0.7)
    JPEG_Q = 72

    frame_idx = 0
    last_boxes = []             # cajas del último frame inferido (para overlay)
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        run_detect = (frame_idx % EVERY_N == 0)
        boxes_for_json = None   # sólo se setea cuando hay inferencia

        if run_detect:
            cur_boxes = []
            with torch.inference_mode():
                # silenciado y tamaño fijo (evita prints "0: 480x640 ...")
                results = model.predict(frame, conf=CONF_TH, imgsz=512, verbose=False)

            for r in results:
                if not hasattr(r, "boxes"): 
                    continue
                for b in r.boxes:
                    conf = float(b.conf[0].detach().cpu().numpy())
                    cls_id = int(b.cls[0].detach().cpu().numpy())
                    clase = model.names[cls_id]

                    if clase != "kiwi" or conf < CONF_TH:
                        continue

                    x1, y1, x2, y2 = map(int, b.xyxy[0].detach().cpu().numpy())
                    w, h = x2 - x1, y2 - y1
                    if w <= 0 or h <= 0:
                        continue
                    ar = (w / h) if h else 0.0
                    if w < MIN_ANCHO or h < MIN_ALTO:
                        continue
                    if not (ASPECT_RATIO_MIN <= ar <= ASPECT_RATIO_MAX):
                        continue

                    cur_boxes.append({
                        "clase": "kiwi",
                        "confianza": conf,
                        "confianza_porcentaje": int(conf * 100),
                        "box": [x1, y1, x2, y2],
                        "ancho": w, "alto": h, "area": w*h,
                    })

            last_boxes = cur_boxes
            boxes_for_json = cur_boxes  # lo que dibujamos es lo que guardamos

        # --- overlay SIEMPRE con las últimas cajas válidas ---
        for kd in last_boxes:
            x1, y1, x2, y2 = kd["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 210, 0), 2)
            cv2.putText(frame, f"kiwi {kd['confianza_porcentaje']}%", (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,210,0), 2)

        # --- persistencia sólo cuando hubo nueva inferencia y hay detecciones ---
        if boxes_for_json:
            nombre_captura = f"captura_video_{frame_idx}.jpg"
            path_captura = os.path.join(CAPTURAS_DIR, nombre_captura)
            cv2.imwrite(path_captura, frame)

            kiwis_detectados = []
            for i, kd in enumerate(boxes_for_json):
                x1, y1, x2, y2 = kd["box"]
                recorte_name = f"Kiwi_recorte_video_{frame_idx}_{i}.jpg"
                cv2.imwrite(os.path.join(RECORTES_DIR, recorte_name), frame[y1:y2, x1:x2])
                kd_out = dict(kd)
                kd_out["archivo_recorte"] = f"recortes/{recorte_name}"
                kiwis_detectados.append(kd_out)

            registro = {
                "timestamp": datetime.now().isoformat(),
                "archivo_captura": f"capturas/{nombre_captura}",
                "origen": "video",
                "nombre_archivo": filename,
                "kiwis_detectados": kiwis_detectados,
                "cantidad_kiwis": len(kiwis_detectados),
            }
            try:
                with open(DB_PATH, "r+", encoding="utf-8") as f:
                    data = json.load(f)
                    data.append(registro)
                    f.seek(0)
                    json.dump(data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print("Error actualizando capturas.json:", e)

        # --- emitir SIEMPRE un frame (fluido) ---
        ok_enc, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q])
        if not ok_enc:
            ok_enc, buffer = cv2.imencode(".jpg", frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        # micro pausa para no saturar CPU pero sin “tirones”
        time.sleep(0.002)
        frame_idx += 1

    cap.release()



@app.get("/stream_video")
def stream_video(filename: str):
    return StreamingResponse(
        generate_frames_from_video(filename),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

#-------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------
# --- Análisis de imágenes ---
@app.get("/analizar_imagenes", response_class=HTMLResponse)
def analizar_imagenes_get(request: Request):
    return templates.TemplateResponse("analizar_imagenes.html",  {"request": request, "now": datetime.now})
@app.get("/resultados_analisis", response_class=HTMLResponse)
def analizar_imagenes_get(request: Request):
    return templates.TemplateResponse("resultados_analisis.html",  {"request": request, "now": datetime.now})

@app.post("/analizar_imagenes", response_class=HTMLResponse)
async def analizar_imagenes_post(
    request: Request,
    files: list[UploadFile] = File(...),
    mm_por_pixel: Optional[str] = Form(None)
):
    try:
        mmpp = float(mm_por_pixel)
        if mmpp <= 0:
            mmpp = 0.04
    except (TypeError, ValueError):
        mmpp = 0.04

    # --- Cargar el modelo YOLOv8 solo una vez ---
    model = getattr(analizar_imagenes_post, "model", None)
    if model is None:
        from ultralytics import YOLO
        model = YOLO(MODELO_PATH_Yolov8)
        analizar_imagenes_post.model = model

    resultados = []
    for file in files:
        nombre_archivo = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        ruta_archivo = os.path.join(ANALISIS_DIR, nombre_archivo)
        with open(ruta_archivo, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        img = cv2.imread(ruta_archivo)
        if img is None:
            continue

        objetos_detectados = []
        #results = model(img)
        results = model(img)
        img_dibujada = img.copy() 
        #results = model(img, conf=0.1, iou=0.2)# iou bajo, conf bajo para debug
        #results[0].show()  
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                clase = model.names[cls_id]
                print(f"Detectado: {clase} | Confianza: {conf:.2f}")
                aspect_ratio = 0
                clase_limpia = clase.lower()
                # Detectar y guardar cualquier clase con confianza suficiente
                if conf > 0.5:  # 
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    ancho = x2 - x1
                    alto = y2 - y1
                    area = ancho * alto
                    aspect_ratio = ancho / alto if alto != 0 else 0
                    if (
                        clase == 'kiwi'
                        and conf > 0.5
                        and ancho >= 20
                        and alto >= 20
                        and ASPECT_RATIO_MIN <= aspect_ratio <= ASPECT_RATIO_MAX
                    ):
                        nombre_recorte = f"kiwi_recorte_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(objetos_detectados)}_{file.filename}"
                        recorte_path = os.path.join(RECORTES_ANALISIS_DIR, nombre_recorte)
                        cv2.imwrite(recorte_path, img[y1:y2, x1:x2])
                        clase_limpia = clase
                        objetos_detectados.append({
                            "clase": clase,
                            "confianza": conf,
                            "bbox": [x1, y1, x2, y2],
                            "recorte": f"recortes_analisis/{nombre_recorte}",
                            "ancho_px": ancho,
                            "alto_px": alto,
                            "area_px": area,
                            "ancho_mm": round(ancho * mmpp, 2),
                            "alto_mm": round(alto * mmpp, 2),
                            "area_mm2": round(area * (mmpp**2), 2)
                        })
                         # Dibujar sobre la imagen de detección
                        cv2.rectangle(img_dibujada, (x1, y1), (x2, y2), (0,255,0), 2)
                        cv2.putText(
                            img_dibujada,
                            f"{clase_limpia} {int(conf*100)}%",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,255,0),
                            3
                        )
        # Guardar la imagen con las detecciones dibujadas
        nombre_deteccion = f"deteccion_{nombre_archivo}"
        ruta_deteccion = os.path.join("resultados_detecciones_kiwi", nombre_deteccion)
        cv2.imwrite(ruta_deteccion, results[0].plot())

        registro = {
            "archivo_original": f"analisis_imagenes/{nombre_archivo}",
            "archivo_deteccion": f"resultados_detecciones_kiwi/{nombre_deteccion}",
            "timestamp": datetime.now().isoformat(),
            "cantidad_objetos": len(objetos_detectados),
            "objetos": objetos_detectados,
            "metadata": {
                "operador": "Simulado",
                "gps": {"lat": -38.2745, "lng": -57.8366},
                "sensor": "Camara Simulada",
                "evento": random.randint(10000, 99999)
            }
        }
        resultados.append(registro)
        # Guardar en JSON
        with open(ANALISIS_JSON, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data.append(registro)
            f.seek(0)
            json.dump(data, f, indent=2, ensure_ascii=False)
    return templates.TemplateResponse(
        "resultados_analisis.html",
        {"request": request, "resultados": resultados, "now": datetime.now}
    )




#------------------ANALSIS DE FLOR CONTEO Y DETERMINAR SI SON HEMBRA O MACHO---------------------------
@app.get("/analizar_imagenes_flor", response_class=HTMLResponse)
def analizar_imagenes_flor_get(request: Request):
    return templates.TemplateResponse("analizar_imagenes_flor.html",  {"request": request, "now": datetime.now})
@app.get("/resultados_analisis_flor", response_class=HTMLResponse)
def analizar_imagenes_flor_get(request: Request):
    return templates.TemplateResponse("resultados_analisis_flor.html",  {"request": request, "now": datetime.now})

@app.post("/analizar_imagenes_flor", response_class=HTMLResponse)
async def analizar_imagenes_flor_post(
    request: Request,
    files: list[UploadFile] = File(...),
    mm_por_pixel: Optional[str] = Form(None)
):
    try:
        mmpp = float(mm_por_pixel)
        if mmpp <= 0:
            mmpp = 0.04
    except (TypeError, ValueError):
        mmpp = 0.04

    total_femeninas = 0
    total_masculinas = 0    
    total_general = 0
    # Inicializar contadores
    conteo_femeninas = 0
    conteo_masculinas = 0
    conteo_total = 0

    # --- Cargar el modelo YOLOv8 solo una vez ---
    modelflor = getattr(analizar_imagenes_flor_post, "model_flor", None)
    if modelflor is None:
        from ultralytics import YOLO
        modelflor = YOLO(MODELO_PATH_Yolov8_Flor)
        analizar_imagenes_flor_post.model_flor = modelflor

    resultados = []
    objetos_detectados = []
    for file in files:
        nombre_archivo = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        ruta_archivo = os.path.join(ANALISIS_DIR_FLOR, nombre_archivo)
        with open(ruta_archivo, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        img = cv2.imread(ruta_archivo)
        if img is None:
            continue
        #results[0].show()  
        img_dibujada = img.copy()  # Crea copia para dibujar detecciones
        objetos_detectados = []
        conteo_femeninas = 0
        conteo_masculinas = 0
        conteo_total = 0

        results = modelflor(img)
        #results[0].show() 
        # Inicializar contadores
        conteo_femeninas = 0
        conteo_masculinas = 0
        conteo_total = 0
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                clase = modelflor.names[cls_id]  # debe ser 'femenina' o 'masculina'
                clase_limpia = clase.lower()
                print(f"Detectado: {clase_limpia} | Confianza: {conf:.2f}")
                aspect_ratio = 0
                if conf > 0.2:  # Usá el umbral bajo para debug
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    ancho = x2 - x1
                    alto = y2 - y1
                    area = ancho * alto
                    aspect_ratio = ancho / alto if alto != 0 else 0
                    if (
                        conf > 0.2
                        and ancho >= 20
                        and alto >= 20
                        and ASPECT_RATIO_MIN <= aspect_ratio <= ASPECT_RATIO_MAX
                    ):
                        
                        conteo_total += 1
                        if clase_limpia == "femenina" or clase_limpia == "flor_femenina":
                            conteo_femeninas += 1
                        elif clase_limpia == "masculina" or clase_limpia == "flor_masculina":
                            conteo_masculinas += 1
                        nombre_recorte = f"flor_recorte_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(objetos_detectados)}_{file.filename}"
                        recorte_path = os.path.join(RECORTES_ANALISIS_DIR_FLOR, nombre_recorte)
                        cv2.imwrite(recorte_path, img[y1:y2, x1:x2])
                        objetos_detectados.append({
                            "clase": clase_limpia,
                            "confianza": conf,
                            "bbox": [x1, y1, x2, y2],
                            "recorte": nombre_recorte,
                            "ancho_px": ancho,
                            "alto_px": alto,
                            "area_px": area,
                            "ancho_mm": round(ancho * mmpp, 2),
                            "alto_mm": round(alto * mmpp, 2),
                            "area_mm2": round(area * (mmpp**2), 2)
                        })
                        # Dibujar sobre la imagen de detección
                        cv2.rectangle(img_dibujada, (x1, y1), (x2, y2), (0,255,0), 2)
                        cv2.putText(
                            img_dibujada,
                            f"{clase_limpia} {int(conf*100)}%",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,255,0),
                            3
                        )
        # Guardar la imagen con las detecciones dibujadas
        nombre_deteccion = f"deteccion_{nombre_archivo}"
        ruta_deteccion = os.path.join("resultados_detecciones_flor", nombre_deteccion)
        cv2.imwrite(ruta_deteccion, results[0].plot())

        registro = {
            "archivo_original": f"analisis_imagenes_flor/{nombre_archivo}",
            "archivo_deteccion": f"resultados_detecciones_flor/{nombre_deteccion}",
            "timestamp": datetime.now().isoformat(),
            "cantidad_objetos": len(objetos_detectados),
            "objetos": objetos_detectados,
            "cantidad_femeninas": conteo_femeninas,
            "cantidad_masculinas": conteo_masculinas,
            "cantidad_total": conteo_total,
            "metadata": {
                "operador": "Simulado",
                "gps": {"lat": -38.2745, "lng": -57.8366},
                "sensor": "Camara Simulada",
                "evento": random.randint(10000, 99999)
            }
        }
        resultados.append(registro)
        # Sumar totales de todas las imágenes procesadas
        total_femeninas = sum(r.get("cantidad_femeninas", 0) for r in resultados)
        total_masculinas = sum(r.get("cantidad_masculinas", 0) for r in resultados)
        total_general = sum(r.get("cantidad_total", 0) for r in resultados)
        # Guardar en JSON
       # ... tu loop de procesamiento de archivos ...
    # Guardar en JSON
    with open(ANALISIS_JSON_FLOR, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data.append(registro)
        f.seek(0)
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Leer el histórico para mostrar los totales acumulados
    with open(ANALISIS_JSON_FLOR, "r", encoding="utf-8") as f:
        historico = json.load(f)
    total_femeninas = sum(int(r.get("cantidad_femeninas", 0) or 0) for r in historico)
    total_masculinas = sum(int(r.get("cantidad_masculinas", 0) or 0) for r in historico)
    total_general = sum(int(r.get("cantidad_total", 0) or 0) for r in historico)

    return templates.TemplateResponse(
        "resultados_analisis_flor.html",
        {
            "request": request,
            "resultados": resultados,  
            "now": datetime.now,
            "total_femeninas": total_femeninas,
            "total_masculinas": total_masculinas,
            "total_general": total_general,
        }
    )
#------------------FIN ANALSIS DE FLOR CONTEO Y DETERMINAR SI SON HEMBRA O MACHO---------------------------


#------------------FIN ANALSIS DE FLOR Y BROTES ---------------------------
@app.get("/analizar_imagenes_flor_brote", response_class=HTMLResponse)
def analizar_imagenes_flor_brote_get(request: Request):
    return templates.TemplateResponse("analizar_imagenes_flor_brote.html",  {"request": request, "now": datetime.now})
@app.get("/resultados_analisis_flor_brote", response_class=HTMLResponse)
def analizar_imagenes_flor_brote_get(request: Request):
    return templates.TemplateResponse("resultados_analisis_flor_brote.html",  {"request": request, "now": datetime.now})

@app.post("/analizar_imagenes_flor_brote", response_class=HTMLResponse)
async def analizar_imagenes_flor_brote_post(
    request: Request,
    files: list[UploadFile] = File(...),
    mm_por_pixel: Optional[str] = Form(None)
):
    try:
        mmpp = float(mm_por_pixel)
        if mmpp <= 0:
            mmpp = 0.04
    except (TypeError, ValueError):
        mmpp = 0.04

    total_brotes = 0
    total_flores = 0    
    total_general = 0
    # Inicializar contadores
    conteo_brotes = 0
    conteo_flores = 0
    conteo_total = 0

    # --- Cargar el modelo YOLOv8 solo una vez ---
    modelflorbrote = getattr(analizar_imagenes_flor_brote_post, "model_flor_brote", None)
    if modelflorbrote is None:
        from ultralytics import YOLO
        # Cargar el modelo de detección de flores y brotes
        modelflorbrote = YOLO(MODELO_PATH_Yolov8_Flor_Brotes)
        # Guardar el modelo en el contexto de la función
        analizar_imagenes_flor_brote_post.model_flor_brote = modelflorbrote
    
    resultados = []
    objetos_detectados = []
    # Procesar cada archivo de imagen
    for file in files:
        nombre_archivo = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        ruta_archivo = os.path.join(ANALISIS_DIR_FLOR_BROTE, nombre_archivo)
        # Guardar el archivo subido en el directcorio de análisis
        with open(ruta_archivo, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # Leer la imagen usando OpenCV
        img = cv2.imread(ruta_archivo)
        # Si la imagen no se pudo leer, continuar con la siguiente
        if img is None:
            continue
        # Crear una copia de la imagen para dibujar las detecciones
        img_dibujada = img.copy()  # Crea copia para dibujar detecciones
        # Inicializar listas y contadores
        objetos_detectados = []
        conteo_brotes = 0
        conteo_flores = 0
        conteo_total = 0
        # Realizar la detección con el modelo YOLOv8
        results = modelflorbrote(img)
        #results[0].show() 
        # Inicializar contadores
        conteo_brotes = 0
        conteo_flores = 0
        conteo_total = 0
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                clase = modelflorbrote.names[cls_id]  # debe ser 'brote' o 'flor'
                # Convertir la clase a minúsculas para comparación
                clase_limpia = clase.lower()
                print(f"Detectado: {clase_limpia} | Confianza: {conf:.2f}")
                aspect_ratio = 0
                if conf > 0.2:  # Usá el umbral bajo para debug
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    ancho = x2 - x1
                    alto = y2 - y1
                    area = ancho * alto
                    aspect_ratio = ancho / alto if alto != 0 else 0
                    if (
                        conf > 0.2
                        and ancho >= 20
                        and alto >= 20
                        and ASPECT_RATIO_MIN <= aspect_ratio <= ASPECT_RATIO_MAX
                    ):
                        
                        conteo_total += 1
                        if clase_limpia == "brote" or clase_limpia == "Brote":
                            conteo_brotes += 1
                        elif clase_limpia == "flor" or clase_limpia == "Flor":
                            conteo_flores += 1
                        nombre_recorte = f"flor_brote_recorte_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(objetos_detectados)}_{file.filename}"
                        recorte_path = os.path.join(RECORTES_ANALISIS_DIR_FLOR_BROTE, nombre_recorte)
                        cv2.imwrite(recorte_path, img[y1:y2, x1:x2])
                        objetos_detectados.append({
                            "clase": clase_limpia,
                            "confianza": conf,
                            "bbox": [x1, y1, x2, y2],
                            "recorte": nombre_recorte,
                            "ancho_px": ancho,
                            "alto_px": alto,
                            "area_px": area,
                            "ancho_mm": round(ancho * mmpp, 2),
                            "alto_mm": round(alto * mmpp, 2),
                            "area_mm2": round(area * (mmpp**2), 2)
                        })
                        # Dibujar sobre la imagen de detección
                        cv2.rectangle(img_dibujada, (x1, y1), (x2, y2), (0,255,0), 2)
                        cv2.putText(
                            img_dibujada,
                            f"{clase_limpia} {int(conf*100)}%",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,255,0),
                            3
                        )
        # Guardar la imagen con las detecciones dibujadas
        nombre_deteccion = f"deteccion_{nombre_archivo}"
        # Especificar la ruta donde se guardará la imagen de detección
        ruta_deteccion = os.path.join(RESULTADOS_DETECCIONES_DIR_FLOR_BROTE, nombre_deteccion)
        cv2.imwrite(ruta_deteccion, results[0].plot())  # Guardar la imagen con las detecciones dibujadas

        # Crear el registro de resultados
        registro = {
            "archivo_original": f"analisis_imagenes_flor_brote/{nombre_archivo}",
            "archivo_deteccion": f"resultados_detecciones_flor_brote/{nombre_deteccion}",
            "timestamp": datetime.now().isoformat(),
            "cantidad_objetos": len(objetos_detectados),
            "objetos": objetos_detectados,
            "cantidad_brotes": conteo_brotes,
            "cantidad_flores": conteo_flores,
            "cantidad_total": conteo_total,
            "metadata": {
                "operador": "Simulado",
                "gps": {"lat": -38.2745, "lng": -57.8366},
                "sensor": "Camara Simulada",
                "evento": random.randint(10000, 99999)
            }
        }
        resultados.append(registro)
        # Sumar totales de todas las imágenes procesadas
        total_brotes = sum(r.get("cantidad_brotes", 0) for r in resultados)
        total_flores = sum(r.get("cantidad_flores", 0) for r in resultados)
        total_general = sum(r.get("cantidad_total", 0) for r in resultados)
        # Guardar en JSON
       # ... tu loop de procesamiento de archivos ...
    # Guardar en JSON
    with open(ANALISIS_JSON_FLOR_BROTE, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data.append(registro)
        f.seek(0)
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Leer el histórico para mostrar los totales acumulados
    with open(ANALISIS_JSON_FLOR_BROTE, "r", encoding="utf-8") as f:
        historico = json.load(f)
    # Sumar los totales de todas las imágenes procesadas
    total_brotes = sum(int(r.get("cantidad_brotes", 0) or 0) for r in historico)
    total_flores = sum(int(r.get("cantidad_flores", 0) or 0) for r in historico)
    total_general = sum(int(r.get("cantidad_total", 0) or 0) for r in historico)
    # Retornar la respuesta con los resultados
    return templates.TemplateResponse(
        "resultados_analisis_flor_brote.html",
        {
            "request": request,
            "resultados": resultados,  
            "now": datetime.now,
            "total_brotes": total_flores,
            "total_flores": total_flores,
            "total_general": total_general,
        }
    )

#------------------FIN ANALSIS DE FLOR Y BROTES ---------------------------

#-------------------------- YEMAS -----------------------------------------
#-------------------------- YEMAS -----------------------------------------
@app.get("/analizar_imagenes_yemas", response_class=HTMLResponse)
def analizar_imagenes_yemas_get(request: Request):
    return templates.TemplateResponse("analizar_imagenes_yemas.html",
                                      {"request": request, "now": datetime.now})

@app.get("/resultados_analisis_yemas", response_class=HTMLResponse)
def resultados_analisis_yemas_get(request: Request):
    return templates.TemplateResponse("resultados_analisis_yemas.html",
                                      {"request": request, "now": datetime.now})

@app.post("/analizar_imagenes_yemas", response_class=HTMLResponse)
async def analizar_imagenes_yemas_post(
    request: Request,
    files: list[UploadFile] = File(...),
    mm_por_pixel: Optional[str] = Form(None)
):
    # mm/px seguro
    try:
        mmpp = float(mm_por_pixel)
        if mmpp <= 0:
            mmpp = 0.04
    except (TypeError, ValueError):
        mmpp = 0.04

    # Cargar el modelo YOLOv8 solo una vez
    model = getattr(analizar_imagenes_yemas_post, "model_yemas", None)
    if model is None:
        from ultralytics import YOLO
        model = YOLO(MODELO_PATH_Yolov8_Yemas)  # <- tu ruta
        analizar_imagenes_yemas_post.model_yemas = model
        # Log rápido de las clases
        try:
            print("[YEMAS] model.names:", model.names)
        except Exception:
            pass

    # Config de inferencia sugerida para objetos chicos
    CONF = 0.25
    IOU  = 0.5
    IMSZ = 1280

    # Filtros anti-ruido
    MIN_W = 8
    MIN_H = 8
    AR_MIN = 0.25
    AR_MAX = 4.0

    resultados = []

    for file in files:
        # Guardar original
        nombre_archivo = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        ruta_archivo = os.path.join(ANALISIS_DIR_YEMAS, nombre_archivo)
        with open(ruta_archivo, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        img = cv2.imread(ruta_archivo)
        if img is None:
            print("[YEMAS] No pude leer la imagen:", ruta_archivo)
            continue

        # Inferencia
        results = model(img, conf=CONF, iou=IOU, imgsz=IMSZ, verbose=False)

        objetos_detectados = []
        conteo_yemas = 0
        conteo_total = 0

        # Nombre real de clase en dataset (aceptamos variantes)
        valid_names = {"yema", "yemas", "bud", "buds"}

        # Para dibujar nosotros también si querés
        img_dib = img.copy()

        h_img, w_img = img.shape[:2]

        for r in results:
            # iterar por índices suele ser más claro con Boxes
            n = 0 if r.boxes is None else len(r.boxes)
            for i in range(n):
                box = r.boxes[i]
                conf = float(box.conf.item())
                cls_id = int(box.cls.item())
                nombre = model.names.get(cls_id, str(cls_id)).lower().strip()

                x1, y1, x2, y2 = map(int, box.xyxy.squeeze().tolist())
                # clamp a la imagen para evitar recortes vacíos
                x1 = max(0, min(x1, w_img - 1))
                y1 = max(0, min(y1, h_img - 1))
                x2 = max(0, min(x2, w_img - 1))
                y2 = max(0, min(y2, h_img - 1))
                if x2 <= x1 or y2 <= y1:
                    continue

                w = x2 - x1
                h = y2 - y1
                ar = w / h

                if conf >= CONF and w >= MIN_W and h >= MIN_H and AR_MIN <= ar <= AR_MAX:
                    # contadores
                    conteo_total += 1
                    if nombre in {"yema", "yemas", "bud", "buds"}:
                        conteo_yemas += 1

                    # recorte
                    rec = img[y1:y2, x1:x2].copy()
                    
                    # --- (opcional) escalar suave si el recorte es muy chico para visualizar ---
                    MIN_VIS_W = 120          # ancho "visible" deseado en la UI
                    if rec.shape[1] < MIN_VIS_W:  # si es muy angosto, lo subo de tamaño para que no se vea “pastoso”
                        scale = MIN_VIS_W / rec.shape[1]
                        rec = cv2.resize(rec, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
                    blur = cv2.GaussianBlur(rec, (0, 0), 0.8)
                    rec = cv2.addWeighted(rec, 1.35, blur, -0.35, 0)

                    rec_name = f"yemas_recorte_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(objetos_detectados)}.png"
                    rec_path = os.path.join(RECORTES_ANALISIS_DIR_YEMAS, rec_name)
                    ok = cv2.imwrite(rec_path, rec, [cv2.IMWRITE_PNG_COMPRESSION, 0]) 
                    
                    """  rec_name = f"yemas_recorte_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(objetos_detectados)}_{file.filename}"
                    rec_path = os.path.join(RECORTES_ANALISIS_DIR_YEMAS, rec_name)
                    ok = cv2.imwrite(rec_path, rec) """
                    if not ok:
                        # si falla, no lo cuentes
                        conteo_total -= 1
                        if nombre in {"yema", "yemas", "bud", "buds"}:
                            conteo_yemas -= 1
                        continue

                    objetos_detectados.append({
                        "clase": nombre,
                        "confianza": conf,
                        "bbox": [x1, y1, x2, y2],
                        "recorte": rec_name,
                        "ancho_px": w,
                        "alto_px": h,
                        "area_px": w * h,
                        "ancho_mm": round(w * mmpp, 2),
                        "alto_mm": round(h * mmpp, 2),
                        "area_mm2": round((w * h) * (mmpp ** 2), 2)
                    })

                    # dibujar (opcional)
                    lw = max(2, int(min(img.shape[:2]) / 400))  # grosor adaptativo
                    cv2.rectangle(img_dib, (x1, y1), (x2, y2), (0, 200, 0), lw)
                    cv2.putText(img_dib, f"{nombre} {int(conf*100)}%", (x1, max(0, y1-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6 + lw*0.1, (0,200,0), max(1, lw))
                    


        # Guardar imagen anotada (con el plot de Ultralytics para asegurar consistencia)
        nombre_deteccion = f"deteccion_{nombre_archivo}"
        ruta_deteccion = os.path.join(RESULTADOS_DETECCIONES_DIR_YEMAS, nombre_deteccion)
        ok_annot = cv2.imwrite(ruta_deteccion, img_dib, [cv2.IMWRITE_JPEG_QUALITY, 100])
        if not ok_annot:
            # fallback a PNG por si falla el codec
            ruta_deteccion = ruta_deteccion.replace(".jpg", ".png")
            cv2.imwrite(ruta_deteccion, img_dib)
        """ try:
            cv2.imwrite(ruta_deteccion, results[0].plot())
        except Exception:
            # fallback a nuestra anotación
            cv2.imwrite(ruta_deteccion, img_dib) """

       # Normalizá los conteos a partir de los objetos realmente guardados
        conteo_total = len(objetos_detectados)
        conteo_yemas = sum(1 for o in objetos_detectados
                        if o.get("clase", "").lower() in {"yema", "yemas", "bud", "buds"})

        registro = {
            "archivo_original":   f"analisis_imagenes_yemas/{nombre_archivo}",
            "archivo_deteccion":  f"resultados_detecciones_yemas/{nombre_deteccion}",
            "timestamp": datetime.now().isoformat(),
            "cantidad_objetos": conteo_total,   # <- igual a len(objetos_detectados)
            "objetos": objetos_detectados,
            "cantidad_yemas": conteo_yemas,
            "cantidad_total": conteo_total,     # <- igual a cantidad_objetos
            "metadata": {
                "operador": "Simulado",
                "gps": {"lat": -38.2745, "lng": -57.8366},
                "sensor": "Camara Simulada",
                "evento": random.randint(10000, 99999)
            }
        }
        resultados.append(registro)

        # Guardar en JSON (por cada imagen)
        try:
            with open(ANALISIS_JSON_YEMAS, "r+", encoding="utf-8") as f:
                data = json.load(f)
                data.append(registro)
                f.seek(0)
                json.dump(data, f, indent=2, ensure_ascii=False)
        except FileNotFoundError:
            with open(ANALISIS_JSON_YEMAS, "w", encoding="utf-8") as f:
                json.dump([registro], f, indent=2, ensure_ascii=False)

    # Leer histórico para totales
    try:
        with open(ANALISIS_JSON_YEMAS, "r", encoding="utf-8") as f:
            historico = json.load(f)
    except FileNotFoundError:
        historico = []

    total_yemas = sum(int(r.get("cantidad_yemas", 0) or 0) for r in historico)
    total_general = sum(int(r.get("cantidad_total", 0) or 0) for r in historico)

    return templates.TemplateResponse(
    "resultados_analisis_yemas.html",
    {
        "request": request,
        "resultados": resultados,
        "now": datetime.now,
        "total_yemas": total_yemas,
        "total_general": total_general,
    }
)


def _to_int(x, default=0):
    try:
        if x is None:
            return default
        if isinstance(x, (int,)):
            return x
        # a veces vienen como "12" o "12.0"
        return int(float(x))
    except Exception:
        return default

try:
    with open(ANALISIS_JSON_YEMAS, "r", encoding="utf-8") as f:
        historico = json.load(f)
except FileNotFoundError:
    historico = []

total_yemas = 0
total_general = 0

for r in historico if isinstance(historico, list) else []:
    # fallback: si no viene cantidad_total, usar cantidad_objetos o len(objetos)
    ct = r.get("cantidad_total", None)
    if ct is None:
        ct = r.get("cantidad_objetos", None)
    if ct is None:
        ct = len(r.get("objetos", []))
    total_general += _to_int(ct)

    cy = r.get("cantidad_yemas", None)
    if cy is None:
        # si no existe, reconstruirlo contando objetos con clase yema/bud
        cy = sum(1 for o in r.get("objetos", [])
                 if str(o.get("clase", "")).lower() in {"yema", "yemas", "bud", "buds"})
    total_yemas += _to_int(cy)


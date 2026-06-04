# utils/video_sampler.py
import logging.config as logging_config
import os
import cv2
import json
import time
import numpy as np
from collections import deque
from datetime import datetime
from ultralytics import YOLO
from skimage.metrics import structural_similarity as ssim
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import logging

log = logging.getLogger("kiwi.sampler")

@dataclass
class SamplerConfig:
    conf_th: float = 0.30
    iou_th:  float = 0.50
    imgsz:   int   = 640
    min_w:   int   = 20
    min_h:   int   = 20
    dt_min:        float = 0.20
    dt_max:        float = 1.00
    change_th:     float = 0.06
    force_every_s: float = 2.0

# Presets
PRESETS: Dict[str, SamplerConfig] = {
    "equilibrado": SamplerConfig(),
    "exhaustivo": SamplerConfig(
        conf_th=0.35, iou_th=0.45, imgsz=640, min_w=24, min_h=24,
        dt_min=0.15, dt_max=0.80, change_th=0.06, force_every_s=1.8
    ),
}

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

# ---------- Utilidades de cambio de escena ----------
def _ssim_change(prev_bgr, curr_bgr, size=(480, 270)):
    prev = cv2.cvtColor(cv2.resize(prev_bgr, size), cv2.COLOR_BGR2GRAY)
    curr = cv2.cvtColor(cv2.resize(curr_bgr, size), cv2.COLOR_BGR2GRAY)
    score = ssim(prev, curr)
    return 1.0 - float(score)  # cambio = 1 - similitud

def _flow_change(prev_bgr, curr_bgr, size=(480, 270)):
    prev = cv2.cvtColor(cv2.resize(prev_bgr, size), cv2.COLOR_BGR2GRAY)
    curr = cv2.cvtColor(cv2.resize(curr_bgr, size), cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev, curr, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1], angleInDegrees=False)
    return float(np.mean(mag))

# ---------- Muestreador adaptativo ----------
class AdaptiveSampler:
    """
    Ajusta dinámicamente cada cuánto muestrear frames en función del cambio de escena.
    Combina SSIM + flujo óptico con EMA e histéresis.
    """
    def __init__(self,
                 dt_min=0.30,      # s: mínimo entre muestras con mucho cambio
                 dt_max=1.20,      # s: máximo entre muestras con poco cambio
                 dt_hard_max=2.50, # s: nunca pasar más de esto sin muestrear
                 ssim_low=0.12,    # cambio alto (1-SSIM) → muestreo denso
                 ssim_high=0.06,   # cambio bajo → muestreo laxo
                 flow_low=0.35,
                 flow_high=0.80,
                 ema_alpha=0.30,
                 cooldown=0.15,
                 use_flow=True):
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_hard_max = dt_hard_max
        self.ssim_low = ssim_low
        self.ssim_high = ssim_high
        self.flow_low = flow_low
        self.flow_high = flow_high
        self.ema_alpha = ema_alpha
        self.cooldown = cooldown
        self.use_flow = use_flow

        self.prev_frame = None
        self.last_sample_ts = None
        self.last_decision_ts = 0.0
        self.change_ema = 0.0
        self.flow_ema = 0.0
        self.dt_target = dt_max

    def _ema(self, val, state):
        return self.ema_alpha * val + (1 - self.ema_alpha) * state

    def _update_dt_target(self):
        fast = (self.change_ema >= self.ssim_low) or (self.flow_ema >= self.flow_high)
        slow = (self.change_ema <= self.ssim_high) and (self.flow_ema <= self.flow_low)
        if fast:
            self.dt_target = self.dt_min
        elif slow:
            self.dt_target = min(self.dt_target * 1.35, self.dt_max)
        else:
            mid = 0.5 * (self.dt_min + self.dt_max)
            self.dt_target = 0.7 * self.dt_target + 0.3 * mid

    def should_sample(self, frame_bgr, ts_ms: int) -> bool:
        ts = ts_ms / 1000.0
        if self.prev_frame is None:
            self.prev_frame = frame_bgr.copy()
            self.last_sample_ts = ts
            self.last_decision_ts = ts
            return True

        # Anti rebote
        if ts - self.last_decision_ts < self.cooldown:
            if ts - self.last_sample_ts >= self.dt_hard_max:
                self.last_decision_ts = ts
                self.last_sample_ts = ts
                self.prev_frame = frame_bgr.copy()
                return True
            return False

        try:
            chg = _ssim_change(self.prev_frame, frame_bgr)
        except Exception:
            chg = 0.0

        flow_val = 0.0
        if self.use_flow:
            try:
                flow_val = _flow_change(self.prev_frame, frame_bgr)
            except Exception:
                flow_val = 0.0

        # EMA
        self.change_ema = self._ema(chg, self.change_ema)
        self.flow_ema   = self._ema(flow_val, self.flow_ema)
        self._update_dt_target()

        elapsed = ts - self.last_sample_ts
        self.last_decision_ts = ts

        if elapsed >= self.dt_target or elapsed >= self.dt_hard_max:
            self.last_sample_ts = ts
            self.prev_frame = frame_bgr.copy()
            return True

        # disparo por pico
        if (chg >= self.ssim_low * 1.6) or (flow_val >= self.flow_high * 1.5):
            self.last_sample_ts = ts
            self.prev_frame = frame_bgr.copy()
            return True

        return False

# ---------- Deduplicación por IoU temporal ----------
def _iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2-inter_x1), max(0, inter_y2-inter_y1)
    inter = iw * ih
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    union = area_a + area_b - inter
    return inter/union if union > 0 else 0.0

class DedupCache:
    def __init__(self, ventana_ms=3000, iou_th=0.55):
        self.win = ventana_ms
        self.th = iou_th
        self.buf = deque()  # {"ts":ms,"cls":str,"bbox":[x1,y1,x2,y2]}

    def is_new(self, det):
        now = det["ts"]
        # limpiar viejos
        while self.buf and now - self.buf[0]["ts"] > self.win:
            self.buf.popleft()
        for d in self.buf:
            if d["cls"] != det["cls"]:
                continue
            if _iou(det["bbox"], d["bbox"]) >= self.th:
                return False
        self.buf.append(det)
        return True

# ---------- Procesador principal ----------
class KiwiVideoSampler:
    """
    Orquesta el muestreo adaptativo + detección YOLO + deduplicación + guardado.
    """
    def __init__(self, model_path, frames_dir, anotados_dir, recortes_dir, log_dir,
             device=0, preset="equilibrado"):
        
        import torch, os

        # ---- NORMALIZAR DEVICE MUY TEMPRANO (evita el AttributeError) ----
        self.device = device
        if isinstance(device, str):
            self.device_str = device
        elif isinstance(device, int):
            self.device_str = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        else:
            self.device_str = "cpu"

        from ultralytics import YOLO
        self.model = YOLO(model_path)

        # ---- MOVER EL MODELO AL DEVICE (una sola vez) ----
        try:
            self.model.to(self.device_str)
        except Exception as e:
            log.warning("[KiwiVideoSampler] No pude mover el modelo al device %s: %s", self.device_str, e)
              
        self.device = device

        self.frames_dir   = frames_dir
        self.anotados_dir = anotados_dir
        self.recortes_dir = recortes_dir
        self.log_dir      = log_dir

        # estado del muestreador
        self.sampler = type("State", (), {})()  # struct simple
        self.sampler.prev_frame = None
        self.sampler.last_sample_ts = None
        self.sampler.last_decision_ts = 0.0
        self.sampler.change_ema = 0.0
        self.sampler.flow_ema = 0.0
        self.sampler.dt_target = None

        # config efectiva
        base_cfg = PRESETS.get(preset, PRESETS["equilibrado"])
        #self.cfg = config or base_cfg
        self._apply_cfg()

        # buffers
        self.last_boxes = []
        self._current_log = {"video": None, "modelo": model_path, "detecciones": []}
    
    def _apply_cfg(self):
        # Normaliza y recorta a rangos válidos
        self.cfg.conf_th = _clamp(float(self.cfg.conf_th), 0.05, 0.95)
        self.cfg.iou_th  = _clamp(float(self.cfg.iou_th),  0.10, 0.95)
        self.cfg.imgsz   = int(self.cfg.imgsz)
        self.cfg.min_w   = int(max(4, self.cfg.min_w))
        self.cfg.min_h   = int(max(4, self.cfg.min_h))
        self.cfg.dt_min  = float(max(0.0, self.cfg.dt_min))
        self.cfg.dt_max  = float(max(self.cfg.dt_min, self.cfg.dt_max))
        self.cfg.change_th     = float(_clamp(self.cfg.change_th, 0.0, 1.0))
        self.cfg.force_every_s = float(max(0.2, self.cfg.force_every_s))
    
    def reset_state(self):
        self.sampler.prev_frame = None
        self.sampler.last_sample_ts = None
        self.sampler.last_decision_ts = 0.0
        self.sampler.change_ema = 0.0
        self.sampler.flow_ema = 0.0
        self.sampler.dt_target = self.cfg.dt_max
        self.last_boxes = []
        self._current_log = {"video": None, "modelo": self._current_log.get("modelo"), "detecciones": []}

    
    def configure(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self.cfg, k) and v is not None:
                setattr(self.cfg, k, v)
        self._apply_cfg()

    def _infer_and_collect(self, frame_bgr, ts_ms: float, base_name: str):
        # 1) Resize opcional manteniendo aspect al lado max cfg.imgsz
        h, w = frame_bgr.shape[:2]
        max_side = int(self.cfg.imgsz)
        if max(h, w) > max_side:
            scale = max_side / float(max(h, w))
            nh, nw = int(round(h * scale)), int(round(w * scale))
            frame_in = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            frame_in = frame_bgr

        # 2) Inferencia YOLO (siempre usando cfg)
        res = self.model(
            frame_in,                  # BGR
            conf=self.cfg.conf_th,
            iou=self.cfg.iou_th,
            imgsz=self.cfg.imgsz,
            verbose=False
        )
        r0 = res[0]

        # DEBUG: ver cuántas cajas devuelve YOLO antes de filtros
        try:
            n_boxes = 0 if not hasattr(r0, "boxes") or r0.boxes is None else len(r0.boxes)
            print(f"[DEBUG] dets_pre_filter={n_boxes}, conf_th={self.cfg.conf_th}, "
                f"iou={self.cfg.iou_th}, imgsz={self.cfg.imgsz}, names={self.model.names}")
        except Exception:
            pass


        # 3) Convertir cajas y aplicar mínimos
        dets = []
        self.last_boxes = []  # para overlay_last()
        for b in (getattr(r0, "boxes", []) or []):
            conf = float(b.conf[0].item())
            cls_id = int(b.cls[0].item())
            x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
            w_box, h_box = x2 - x1, y2 - y1
            if w_box < self.cfg.min_w or h_box < self.cfg.min_h:
                continue
            clase = self.model.names[cls_id]
            dets.append({
                "clase": clase,
                "conf": conf,
                "bbox": [x1, y1, x2, y2],
                "recorte": None
            })
            self.last_boxes.append((x1, y1, x2, y2, clase, conf))

        # 4) Dibujar anotación y guardar (en frames_anotados)
        annotated = frame_in.copy()
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(annotated, f'{d["clase"]} {int(d["conf"]*100)}%',
                        (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # 👇 **ACTUALIZAR EL OVERLAY**
        self.last_boxes = [
            (d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3], d["clase"], d["conf"])
            for d in dets
        ]

        # nombre de archivo basado en ts_ms
        fname = f"{base_name}_{int(ts_ms)}.jpg"
        anotado_path = os.path.join(self.anotados_dir, fname)
        cv2.imwrite(anotado_path, annotated)

        # 5) Guardar recortes (opcional)
        for idx, d in enumerate(dets):
            x1, y1, x2, y2 = d["bbox"]
            crop = frame_in[y1:y2, x1:x2]
            crop_name = f"{base_name}_{int(ts_ms)}_{idx}.jpg"
            crop_path = os.path.join(self.recortes_dir, crop_name)
            if crop.size > 0:
                cv2.imwrite(crop_path, crop)
                d["recorte"] = crop_name

        return dets, os.path.basename(anotado_path)


    def step(self, frame_bgr, ts_ms, video_name="video"):
        """
        Llamar por cada frame. Si corresponde muestrear:
         - guarda el frame base
         - corre detección
         - guarda recortes y anotado
         - retorna un 'record' para log/DB
        """
        # preparar nombres
        ts_str = f"{int(ts_ms):09d}ms"
        base_name = f"{os.path.splitext(video_name)[0]}_{ts_str}"

        # decidir si muestrear
        if not self._should_sample(frame_bgr, ts_ms):
            return None

        # guardar frame base (opcional)
        frame_path = os.path.join(self.frames_dir, f"{base_name}.jpg")
        cv2.imwrite(frame_path, frame_bgr)

        # inferencia + guardados
        dets, annotated_path = self._infer_and_collect(frame_bgr, ts_ms, base_name)

        record = {
            "timestamp_ms": int(ts_ms),
            "frame_jpg": os.path.basename(frame_path),
            "frame_anotado_jpg": os.path.basename(annotated_path) if annotated_path else None,
            "objetos": dets
        }
        return record

    def overlay_last(self, frame_bgr):
        """
        Dibuja sobre 'frame_bgr' las últimas cajas detectadas (no guarda nada).
        Útil para streaming fluido.
        """
        if not self.last_boxes:
            return frame_bgr
        out = frame_bgr.copy()
        for (x1,y1,x2,y2,clase,conf) in self.last_boxes:
            cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(out, f"{clase} {conf*100:.0f}%", (x1, max(0, y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        return out

    def append_log(self, video_name, record, save_every=10):
        """
        Acumula registros y guarda a disco cada 'save_every' muestras.
        """
        if self._current_log["video"] is None:
            self._current_log["video"] = video_name
        self._current_log["detecciones"].append(record)
        if len(self._current_log["detecciones"]) % save_every == 0:
            self.flush_log(video_name)

    def flush_log(self, video_name):
        ruta_log = os.path.join(self.log_dir, f"{os.path.splitext(video_name)[0]}_muestreo.json")
        with open(ruta_log, "w", encoding="utf-8") as f:
            json.dump(self._current_log, f, indent=2, ensure_ascii=False)
            
    def _should_sample(self, frame_bgr, ts_ms: float) -> bool:
        cfg = self.cfg
        s = self.sampler

        t = float(ts_ms) / 1000.0

        # warmup
        if s.last_sample_ts is None or s.prev_frame is None:
            s.prev_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            s.last_sample_ts = t
            return True

        dt = t - s.last_sample_ts
        if dt < cfg.dt_min:
            return False

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        try:
            from skimage.metrics import structural_similarity as ssim
            score = ssim(s.prev_frame, gray)
            change = 1.0 - float(score)
        except Exception:
            diff = cv2.absdiff(s.prev_frame, gray)
            change = float(diff.mean()) / 255.0

        s.prev_frame = gray

        if t - s.last_sample_ts >= cfg.force_every_s:
            s.last_sample_ts = t
            return True

        if change >= cfg.change_th:
            s.last_sample_ts = t
            return True

        if dt >= cfg.dt_max:
            s.last_sample_ts = t
            return True

        return False


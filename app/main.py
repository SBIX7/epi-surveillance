"""
app/main.py — Serveur FastAPI SmartSafety Vision
Routes : homepage, démarrage flux, upload vidéo/modèle, MJPEG stream.
"""

import json
import time
from pathlib import Path
from typing import Generator, Optional
import threading

import cv2
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.vision import ModelLoadError, PPE_MODEL_HF, SafetyGearEngine

# ── Chemins ──────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
DEMO_VIDEO = BASE_DIR / "assets" / "demo_chantier.mp4"
CONFIG_DIR = BASE_DIR / ".config"
MODEL_CONFIG_FILE = CONFIG_DIR / "model_config.json"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

def get_asset_version() -> int:
    """Retourne un numéro de version basé sur la date de modification des assets statiques."""
    try:
        css_file = BASE_DIR / "static" / "css" / "styles.css"
        js_file = BASE_DIR / "static" / "js" / "script.js"
        return int(max(css_file.stat().st_mtime, js_file.stat().st_mtime))
    except Exception:
        return int(time.time())

# ── Application FastAPI ───────────────────────────────────────────────────
app = FastAPI(
    title="SmartSafety Vision API",
    description="Détection EPI en temps réel — YOLOv8 + OpenCV",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ── Moteur de vision (singleton) ──────────────────────────────────────────
# Lance automatiquement : keremberke/yolov8n-ppe-detection → yolov8n.pt
# Charge en priorité le dernier modèle qui a été chargé

def get_persisted_model_path() -> Optional[str]:
    """Récupère le chemin du modèle persisté, s'il existe."""
    try:
        if MODEL_CONFIG_FILE.exists():
            with open(MODEL_CONFIG_FILE, 'r') as f:
                config = json.load(f)
                model_path = config.get("model_path")
                if model_path and Path(model_path).exists():
                    return model_path
    except Exception as e:
        print(f"[SmartSafety] Erreur lecture config modèle : {e}")
    return None

def save_model_path(model_path: str) -> None:
    """Sauvegarde le chemin du modèle chargé pour persistance."""
    try:
        config = {"model_path": str(model_path)}
        with open(MODEL_CONFIG_FILE, 'w') as f:
            json.dump(config, f)
        print(f"[SmartSafety] Modèle sauvegardé en config : {model_path}")
    except Exception as e:
        print(f"[SmartSafety] Erreur écriture config modèle : {e}")

# Charger le modèle persisté s'il existe
# NOTE: Désactivé pour utiliser le modèle PPE HuggingFace qui est mieux entraîné
# persisted_model = get_persisted_model_path()
# vision = SafetyGearEngine(model_path=persisted_model if persisted_model else None)
vision = SafetyGearEngine(model_path=None)

if vision._epi_classes_available:
    print(f"[SmartSafety] Classes EPI disponibles : {[v for v in vision.class_names.values()]}")
else:
    print("[SmartSafety] AVERTISSEMENT : aucune classe EPI dans le modele actuel.")
    print("[SmartSafety] Uploadez un modele PPE .pt via la sidebar pour activer la detection.")

# ── État d'alerte ──────────────────────────────────────────────────────────
alert_state = {"active": False}

# ── État partagé du flux vidéo ────────────────────────────────────────────
stream_state: dict = {
    "source":     "camera",   # "camera" | "upload"
    "video_path": None,       # Path vers le fichier vidéo (si upload/démo)
    "capture":    None,       # cv2.VideoCapture actif
}


class VideoSourceError(Exception):
    """Levée quand la source vidéo est introuvable ou inouvrable."""


# ── Gestion de la capture ─────────────────────────────────────────────────
class ThreadedCamera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise VideoSourceError("Impossible d'ouvrir la source vidéo.")
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            if grabbed:
                self.grabbed, self.frame = grabbed, frame
            else:
                time.sleep(0.01)

    def read(self):
        return self.grabbed, self.frame

    def release(self):
        self.stopped = True
        self.cap.release()
        self.thread.join(timeout=1.0)
        
    def isOpened(self):
        return self.cap.isOpened()
        
    def set(self, propId, value):
        return self.cap.set(propId, value)

def release_capture() -> None:
    """Libère proprement la capture OpenCV courante."""
    cap = stream_state.get("capture")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    stream_state["capture"] = None


def get_capture() -> cv2.VideoCapture:
    """
    Retourne la capture active ; en crée une nouvelle si nécessaire.

    Raises
    ------
    VideoSourceError si la source ne peut pas être ouverte.
    """
    cap = stream_state.get("capture")
    if cap is not None and cap.isOpened():
        return cap

    if stream_state["source"] == "camera":
        cap = ThreadedCamera(0)
    elif stream_state["video_path"]:
        cap = cv2.VideoCapture(str(stream_state["video_path"]))
    else:
        raise VideoSourceError("Aucune source vidéo définie.")

    if not cap.isOpened():
        raise VideoSourceError(
            "Impossible d'ouvrir la source vidéo. "
            "Vérifiez que la caméra est disponible ou que le fichier existe."
        )

    stream_state["capture"] = cap
    return cap


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, summary="Page principale du dashboard")
def homepage(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "asset_version": get_asset_version(),
        },
    )


@app.get("/alert", summary="État d'alerte pour violations EPI")
def get_alert() -> JSONResponse:
    """Retourne True si des violations EPI sont détectées dans la frame actuelle."""
    return JSONResponse({"alert": alert_state["active"]})


@app.get("/debug", summary="Infos de debug du système")
def get_debug() -> JSONResponse:
    """Retourne l'état actuel du modèle et de l'alerte pour debug."""
    model_info = vision.get_model_info()
    return JSONResponse({
        "model_info": model_info,
        "alert_active": alert_state["active"],
        "stream_state": {
            "source": stream_state["source"],
            "has_video_path": stream_state["video_path"] is not None,
        }
    })


@app.post(
    "/start_camera",
    summary="Démarrer le flux depuis la webcam locale",
    response_description="URL du flux MJPEG",
)
async def start_camera() -> JSONResponse:
    """
    Libère toute capture active et initialise la webcam (index 0).
    Retourne un JSON avec l'URL du flux `/video_feed`.
    """
    release_capture()
    stream_state["source"]     = "camera"
    stream_state["video_path"] = None
    try:
        get_capture()
    except VideoSourceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse({
        "message":    "Flux caméra locale prêt.",
        "stream_url": "/video_feed",
    })


@app.post(
    "/start_demo",
    summary="Démarrer la vidéo de démonstration",
    response_description="URL du flux MJPEG",
)
async def start_demo() -> JSONResponse:
    """
    Coupe tout flux en cours et lance `assets/demo_chantier.mp4`.
    Retourne HTTP 404 si la vidéo de démo est absente.
    """
    release_capture()

    if not DEMO_VIDEO.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Vidéo de démonstration introuvable : {DEMO_VIDEO}. "
                "Placez un fichier demo_chantier.mp4 dans le dossier assets/."
            ),
        )

    stream_state["source"]     = "upload"
    stream_state["video_path"] = DEMO_VIDEO

    try:
        get_capture()
    except VideoSourceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse({
        "message":    "Vidéo de démonstration lancée.",
        "stream_url": "/video_feed",
    })


@app.post(
    "/upload_video",
    summary="Uploader une vidéo MP4 et l'analyser",
    response_description="URL du flux MJPEG",
)
async def upload_video(file: UploadFile = File(...)) -> JSONResponse:
    """
    Reçoit un fichier MP4, le sauvegarde dans `static/uploads/`,
    et démarre le flux d'analyse dessus.
    """
    if file.content_type != "video/mp4":
        raise HTTPException(
            status_code=400,
            detail="Seuls les fichiers MP4 sont acceptés (video/mp4).",
        )

    timestamp   = int(time.time())
    target_path = UPLOAD_DIR / f"video_{timestamp}_{file.filename}"
    content     = await file.read()
    target_path.write_bytes(content)

    release_capture()
    stream_state["source"]     = "upload"
    stream_state["video_path"] = target_path

    try:
        get_capture()
    except VideoSourceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse({
        "message":    f"Vidéo « {file.filename} » uploadée et prête à l'analyse.",
        "stream_url": "/video_feed",
    })


@app.post(
    "/upload_model",
    summary="Charger un modèle YOLOv8 personnalisé (.pt)",
    response_description="Confirmation du modèle chargé",
)
async def upload_model(file: UploadFile = File(...)) -> JSONResponse:
    """
    Reçoit un fichier .pt Ultralytics, le sauvegarde et recharge le moteur.
    """
    if not file.filename.lower().endswith(".pt"):
        raise HTTPException(
            status_code=400,
            detail="Le modèle doit être un fichier .pt (Ultralytics YOLOv8).",
        )

    timestamp   = int(time.time())
    target_path = UPLOAD_DIR / f"model_{timestamp}_{file.filename}"
    content     = await file.read()
    
    # Verify file is not empty
    if not content:
        raise HTTPException(
            status_code=400,
            detail="Le fichier uploadé est vide. Vérifiez que le fichier .pt est valide.",
        )
    
    try:
        target_path.write_bytes(content)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la sauvegarde du fichier : {str(exc)}",
        ) from exc

    try:
        vision.load_model(str(target_path))
        # Sauvegarder le chemin du modèle chargé
        save_model_path(str(target_path))
    except ModelLoadError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du chargement du modèle : {type(exc).__name__} - {str(exc)}",
        ) from exc

    info = vision.get_model_info()
    warnings = []
    if not info["epi_classes_available"]:
        warnings.append(
            f"Aucune classe EPI reconnue parmi : {info['all_classes']}. "
            "Ce modele ne peut pas détecter les EPI (casque, gilet, gants, etc.)."
        )
    elif not info["person_detected"]:
        # Modèle EPI-only valide (ex: best0.pt) — pas une erreur
        warnings.append(
            "Modèle EPI-only détecté : pas de classe 'personne'. "
            "Les EPI seront annotés directement sans association à une personne."
        )

    msg = (
        f"Modele '{file.filename}' charge ({info['total_classes']} classes"
        + (f", personnes: {info['person_classes']}" if info['person_detected'] else "")
        + (f", EPI: {info['epi_classes']}" if info['epi_classes_available'] else "")
        + ")."
    )
    return JSONResponse({
        "message":   msg,
        "model_name": file.filename,
        "warnings":  warnings,
        **info,
    })


@app.post(
    "/stop_stream",
    summary="Arrêter le flux vidéo en cours",
    response_description="Confirmation d'arrêt du flux",
)
async def stop_stream() -> JSONResponse:
    """Arrête proprement la capture en cours et libère la caméra."""
    release_capture()
    return JSONResponse({"message": "Flux vidéo arrêté."})


# ── Générateur MJPEG ──────────────────────────────────────────────────────
def _generate_frames() -> Generator[bytes, None, None]:
    """
    Générateur Python qui :
      1. Lit une frame depuis la capture active.
      2. Passe la frame dans `vision.predict_and_annotate()` (tous les 2 frames pour optimiser).
      3. Encode en JPEG et yield le chunk MJPEG.
      4. Libère la capture en fin de boucle (vidéo terminée ou déconnexion).
    """
    try:
        capture = get_capture()
    except VideoSourceError as exc:
        # Impossible d'avoir une HTTPException dans un générateur streamé ;
        # on log et on sort proprement.
        print(f"[SmartSafety] VideoSourceError : {exc}")
        return

    last_annotated = None
    frame_count = 0
    skip_frames = 2  # Traiter 1 frame sur 2 pour doubler la fluidité

    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            # Fin de fichier vidéo → rebobinage pour la démo
            if stream_state["source"] == "upload" and stream_state["video_path"]:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = capture.read()
                if not ok:
                    break
            else:
                break

        # Optimisation : traiter seulement tous les 'skip_frames' frames
        if frame_count % skip_frames == 0:
            try:
                last_annotated, has_violations = vision.predict_and_annotate(frame)
                alert_state["active"] = has_violations
            except Exception as exc:
                print(f"[SmartSafety] Erreur annotation : {exc}")
                last_annotated = frame.copy()
                alert_state["active"] = False
        else:
            # Réutiliser la dernière annotation pour les frames sautés
            if last_annotated is None:
                last_annotated = frame.copy()
                alert_state["active"] = False

        annotated = last_annotated
        frame_count += 1

        ok2, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok2:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + encoded.tobytes()
            + b"\r\n"
        )
        # Supprimer le sleep pour maximiser la fluidité (inference limite déjà)
        # time.sleep(0.02)

    release_capture()


@app.get(
    "/video_feed",
    summary="Flux MJPEG annoté en temps réel",
    response_class=StreamingResponse,
)
def video_feed() -> StreamingResponse:
    """Retourne le flux MJPEG (`multipart/x-mixed-replace`) annoté par YOLO."""
    return StreamingResponse(
        _generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

"""
app/main.py — Serveur FastAPI SmartSafety Vision
Routes : homepage, démarrage flux, upload vidéo/modèle, MJPEG stream.
"""

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
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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
vision = SafetyGearEngine(model_path=None)

if vision._epi_classes_available:
    print(f"[SmartSafety] Classes EPI disponibles : {[v for v in vision.class_names.values()]}")
else:
    print("[SmartSafety] AVERTISSEMENT : aucune classe EPI dans le modele actuel.")
    print("[SmartSafety] Uploadez un modele PPE .pt via la sidebar pour activer la detection.")

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
    return templates.TemplateResponse(request, "index.html")


@app.get("/model_info", summary="Infos sur le modele YOLO actif")
def model_info() -> JSONResponse:
    """
    Retourne les classes du modèle actif :
    - person_classes : classes reconnues comme 'personne'
    - epi_classes    : classes EPI (casque, gilet...)
    - other_classes  : reste
    Utile pour diagnostiquer un modèle custom.
    """
    return JSONResponse(vision.get_model_info())


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
    target_path.write_bytes(content)

    try:
        vision.load_model(str(target_path))
    except ModelLoadError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

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


# ── Générateur MJPEG ──────────────────────────────────────────────────────
def _generate_frames() -> Generator[bytes, None, None]:
    """
    Générateur Python qui :
      1. Lit une frame depuis la capture active.
      2. Passe la frame dans `vision.predict_and_annotate()`.
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

        try:
            annotated = vision.predict_and_annotate(frame)
        except Exception as exc:
            print(f"[SmartSafety] Erreur annotation : {exc}")
            annotated = frame.copy()

        ok2, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok2:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + encoded.tobytes()
            + b"\r\n"
        )
        # Léger délai pour éviter de saturer le navigateur avec un flux trop rapide
        # particulièrement lors de la lecture d'un fichier vidéo pré-enregistré.
        time.sleep(0.01)

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

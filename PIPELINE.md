# PIPELINE — SmartSafety Vision

> Documentation technique du système de détection EPI en temps réel.  
> Architecture, algorithmes, et flux d'utilisation.

---

## Table des matières

1. [Architecture globale](#1-architecture-globale)
2. [Pipeline de traitement vidéo](#2-pipeline-de-traitement-vidéo)
3. [Logique métier — Algorithme d'association EPI ↔ Personne](#3-logique-métier--algorithme-dassociation-epi--personne)
4. [Flux d'utilisation — Sources vidéo](#4-flux-dutilisation--sources-vidéo)
5. [Structure du projet](#5-structure-du-projet)
6. [Lancement du serveur](#6-lancement-du-serveur)

---

## 1. Architecture globale

```
┌─────────────────────────────────────────────────────────────────┐
│                         NAVIGATEUR WEB                          │
│                                                                 │
│  ┌──────────────────────┐    ┌──────────────────────────────┐  │
│  │   Dashboard HTML/CSS │    │  <img> MJPEG Stream         │  │
│  │   (controls, toasts) │    │  src="/video_feed"          │  │
│  └──────────┬───────────┘    └──────────────┬───────────────┘  │
│             │  Fetch POST (JSON)             │  multipart/mjpeg │
└─────────────┼────────────────────────────────┼──────────────────┘
              │                                │
              ▼                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FASTAPI  (uvicorn)                         │
│                                                                 │
│  Routes JSON          │  Route streaming                       │
│  ─────────────────    │  ──────────────────────────────────    │
│  POST /start_camera   │  GET /video_feed                       │
│  POST /start_demo     │    └─ générateur Python (yield)        │
│  POST /upload_video   │         │                              │
│  POST /upload_model   │         ▼                              │
│                       │  ┌──────────────────────────────────┐  │
│                       │  │      SafetyGearEngine            │  │
│                       │  │  (app/vision.py)                 │  │
│                       │  │                                  │  │
│                       │  │  predict_and_annotate(frame)     │  │
│                       │  └──────────────┬───────────────────┘  │
│                       │                 │                       │
│                       │                 ▼                       │
│                       │  ┌──────────────────────────────────┐  │
│                       │  │   Ultralytics YOLOv8 (.pt)       │  │
│                       │  │   Inférence sur GPU/CPU          │  │
│                       │  └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  OpenCV (cv2)                                                   │
│  • VideoCapture : lecture frames depuis webcam ou fichier MP4  │
│  • imencode : compression JPEG pour le flux MJPEG             │
│  • rectangle / putText : dessin des bounding boxes            │
└─────────────────────────────────────────────────────────────────┘
```

### Composants

| Composant | Rôle |
|---|---|
| **FastAPI** | Serveur HTTP asynchrone, routes JSON + streaming MJPEG |
| **Uvicorn** | Serveur ASGI (ASGI = interface async Python ↔ HTTP) |
| **OpenCV** | Lecture vidéo, encodage JPEG, dessin des annotations |
| **Ultralytics YOLOv8** | Modèle de détection d'objets (personnes + EPI) |
| **HTML/CSS/JS** | Dashboard interactif, système de toasts, drag-and-drop |

---

## 2. Pipeline de traitement vidéo

Chaque frame passe par les étapes suivantes en `app/main.py` → `app/vision.py` :

```
Source vidéo
  │  (webcam idx 0  OU  fichier MP4)
  │
  ▼
cv2.VideoCapture.read()
  │  → frame BGR (numpy.ndarray H×W×3)
  │
  ▼
SafetyGearEngine.predict_and_annotate(frame)
  │
  ├─ 1. INFÉRENCE YOLO
  │     model(frame, conf=0.25, imgsz=640)
  │     → liste de bounding boxes + labels + scores
  │
  ├─ 2. TRI DES DÉTECTIONS
  │     ┌─────────────┬─────────────────┐
  │     │  "person"   │  label EPI ?    │
  │     │  → persons  │  → gears        │
  │     └─────────────┴─────────────────┘
  │
  ├─ 3. ASSOCIATION EPI ↔ PERSONNE  (voir §3)
  │
  ├─ 4. ANNOTATION OpenCV
  │     • Rectangle ORANGE pour chaque EPI
  │     • Rectangle VERT  si EPI complet
  │     • Rectangle ROUGE si EPI manquant
  │     • Fond coloré derrière chaque texte
  │
  └─ 5. Retour : frame annotée (np.ndarray)
         │
         ▼
cv2.imencode(".jpg", annotated, quality=85)
         │
         ▼
yield  b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + bytes + b"\r\n"
         │
         ▼
  Navigateur reçoit le flux MJPEG
  et rafraîchit l'<img> automatiquement
```

### Format MJPEG (multipart)

Le protocole **MJPEG** consiste à envoyer une série de JPEG séparés par une
*boundary* dans une réponse HTTP qui ne se ferme jamais :

```
Content-Type: multipart/x-mixed-replace; boundary=frame

--frame
Content-Type: image/jpeg

<bytes JPEG frame 1>
--frame
Content-Type: image/jpeg

<bytes JPEG frame 2>
...
```

Le navigateur reconnaît ce format nativement dans une balise `<img>` et affiche
chaque nouvelle image dès sa réception, créant l'illusion d'une vidéo fluide.

---

## 3. Logique métier — Algorithme d'association EPI ↔ Personne

### Problème

YOLOv8 détecte *tous* les objets dans la frame de manière indépendante.
Il retourne, par exemple :
- 2 bounding boxes **"person"**
- 1 bounding box **"helmet"**
- 1 bounding box **"vest"**

Il faut savoir **quel casque appartient à quelle personne**.

### Solution : Ratio d'intersection

On mesure le chevauchement entre la bounding box d'un EPI et celle d'une personne.

```
  Bounding box PERSONNE         Bounding box CASQUE
  ┌───────────────────┐         ┌───────┐
  │                   │         │       │
  │      ┌────────────┼─────────┤       │
  │      │            │  Zone   │       │
  │      │  INTERSEC  │ d'inter-│       │
  │      │            │ section │       │
  │      └────────────┼─────────┘       │
  │                   │
  └───────────────────┘
```

**Formule :**

```
                    Aire(Intersection)
overlap_ratio  =  ────────────────────
                  Aire(Personne)
```

**Implémentation (`vision.py`) :**

```python
ix1 = max(person_box[0], gear_box[0])   # x gauche intersection
iy1 = max(person_box[1], gear_box[1])   # y haut  intersection
ix2 = min(person_box[2], gear_box[2])   # x droit intersection
iy2 = min(person_box[3], gear_box[3])   # y bas   intersection

inter_w    = max(0, ix2 - ix1)
inter_h    = max(0, iy2 - iy1)
inter_area = inter_w * inter_h

person_area   = max(1, (person_box[2]-person_box[0]) * (person_box[3]-person_box[1]))
overlap_ratio = inter_area / person_area
```

### Pourquoi ce ratio (et non l'IoU classique) ?

L'IoU (Intersection over Union) divise par l'aire de l'*union*, ce qui
pénalise les petits EPI. Un casque représente ~3–5 % de la surface d'un ouvrier ;
son IoU serait négligeable même s'il est bien porté.

En divisant par l'**aire de la personne**, on mesure la proportion de la
bounding-box de l'ouvrier couverte par l'EPI. Un seuil de **2 %** est
suffisant pour associer un casque situé sur la tête.

```
overlap_ratio > GEAR_OVERLAP_THRESHOLD (0.02)
  → EPI considéré comme porté par cette personne ✓
```

### Exemple visuel

```
Personne A (grande bbox)        Personne B (bbox droite)
┌────────────────┐              ┌──────────────┐
│   [Casque A]   │              │  [Casque B]  │
│ overlap = 4 %  │              │ overlap = 3 %│
│   ✓ EPI OK     │              │   ✓ EPI OK   │
│                │              │              │
│  [Gilet]       │              │   (pas de    │
│  overlap = 12% │              │    gilet)    │
│                │              │   ✗ MANQUANT │
└────────────────┘              └──────────────┘
```

### Groupes d'EPI configurables

```python
REQUIRED_GEAR_GROUPS = {
    "Casque":                 ["helmet", "hardhat", "headgear", "hard-hat"],
    "Gilet haute visibilité": ["vest", "jacket", "safety", "hi-vis", "hivis"],
}
```

Si un groupe est absent de `found_groups` → affiché en rouge avec le message
`MANQUANT : <Groupe>`.

---

## 4. Flux d'utilisation — Sources vidéo

### 4.1 Webcam locale

```
[Bouton "Caméra locale"]
       │
       ▼  POST /start_camera
FastAPI → release_capture()
        → cv2.VideoCapture(0)
        → JSONResponse { stream_url: "/video_feed" }
       │
       ▼
Frontend → <img src="/video_feed">
```

> **Prérequis :** Une webcam accessible sur l'index 0 du système.

---

### 4.2 Upload MP4

```
[Zone drag & drop / sélecteur fichier]
       │  Fichier .mp4 sélectionné
       ▼  POST /upload_video  (multipart/form-data)
FastAPI → validation Content-Type (video/mp4)
        → sauvegarde dans static/uploads/video_<ts>_<nom>.mp4
        → release_capture()
        → stream_state["video_path"] = chemin sauvegardé
        → JSONResponse { message, stream_url }
       │
       ▼
Frontend → activateStream() → <img src="/video_feed?<cache-bust>">
```

---

### 4.3 Mode Démo (`assets/demo_chantier.mp4`)

```
[Bouton "Vidéo Démo"]
       │
       ▼  POST /start_demo
FastAPI → release_capture()
        → vérifie que assets/demo_chantier.mp4 existe (404 sinon)
        → stream_state["video_path"] = DEMO_VIDEO
        → cv2.VideoCapture(str(DEMO_VIDEO))
        → JSONResponse { stream_url: "/video_feed" }
       │
       ▼  GET /video_feed
Générateur → rebobinage automatique en fin de fichier
             (capture.set(CAP_PROP_POS_FRAMES, 0))
             → lecture en boucle infinie
```

---

### 4.4 Modèle personnalisé

```
[Zone upload modèle .pt]
       │
       ▼  POST /upload_model
FastAPI → validation extension .pt
        → sauvegarde dans static/uploads/model_<ts>_<nom>.pt
        → vision.load_model(chemin)
           └─ YOLO(model_path)  ← rechargement à chaud
        → JSONResponse { message, model_name }
```

> Le flux vidéo en cours n'est **pas** interrompu lors du changement de modèle.
> Le nouveau modèle est utilisé dès la frame suivante.

---

## 5. Structure du projet

```
mehdi_tp_fr/
│
├── app/
│   ├── __init__.py
│   ├── main.py          # Serveur FastAPI, routes, générateur MJPEG
│   └── vision.py        # SafetyGearEngine : inférence + annotation
│
├── assets/
│   └── demo_chantier.mp4  # Vidéo de démonstration (à fournir)
│
├── static/
│   ├── css/
│   │   └── styles.css   # Design system dark mode
│   ├── js/
│   │   └── script.js    # Logique frontend (toasts, drag&drop, fetch)
│   └── uploads/         # Vidéos et modèles uploadés (auto-créé)
│
├── templates/
│   └── index.html       # Dashboard SaaS industriel
│
├── pyproject.toml
├── requirements.txt
└── PIPELINE.md          # Ce fichier
```

---

## 6. Lancement du serveur

### Avec `uv` (recommandé)

```bash
# Installation des dépendances
uv sync

# Démarrage du serveur de développement
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Avec pip classique

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Accès

| URL | Description |
|---|---|
| `http://localhost:8000/` | Dashboard principal |
| `http://localhost:8000/video_feed` | Flux MJPEG brut |
| `http://localhost:8000/docs` | Documentation OpenAPI (Swagger UI) |
| `http://localhost:8000/redoc` | Documentation ReDoc |

---

> **SmartSafety Vision** — Système de surveillance EPI par IA  
> Stack : FastAPI · Ultralytics YOLOv8 · OpenCV · Vanilla JS

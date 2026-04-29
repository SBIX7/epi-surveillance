# PIPELINE — SmartSafety Vision

> Documentation technique du projet de surveillance EPI.
> Architecture, traitement vidéo, logiques métier et intégration GitHub.

---

## Table des matières

1. [Architecture globale](#1-architecture-globale)
2. [Pipeline de traitement vidéo](#2-pipeline-de-traitement-vidéo)
3. [Logique métier — association EPI ↔ personne](#3-logique-métier--association-epi-↔-personne)
4. [Flux d'utilisation](#4-flux-dutilisation)
5. [Structure du projet](#5-structure-du-projet)
6. [Lancement du serveur](#6-lancement-du-serveur)
7. [Préparation GitHub](#7-préparation-github)

---

## 1. Architecture globale

Le projet s’organise en trois couches principales :

- **Frontend Web** : interface d’upload, contrôle du flux et affichage du MJPEG.
- **Backend FastAPI** : routes HTTP, gestion des sources vidéo et route MJPEG.
- **Moteur de vision** : inférence YOLOv8 et annotation OpenCV.

### Architecture fonctionnelle

```
NAVIGATEUR WEB
  ├─ interface utilisateur (templates/index.html)
  ├─ actions utilisateur
  └─ flux MJPEG

FASTAPI (app/main.py)
  ├─ /
  ├─ /start_camera
  ├─ /start_demo
  ├─ /upload_video
  ├─ /upload_model
  ├─ /video_feed
  ├─ /alert
  └─ /debug

SAFETYGEAR ENGINE (app/vision.py)
  ├─ chargement du modèle YOLO
  ├─ inférence par frame
  ├─ filtrage des classes
  ├─ association EPI ↔ personne
  └─ annotation OpenCV

YOLOv8 (.pt)
  ├─ détection des personnes
  └─ détection des objets EPI
```

### Composants clés

- **FastAPI** : API REST et streaming MJPEG.
- **Uvicorn** : serveur ASGI.
- **OpenCV** : capture vidéo, dessin, encodage JPEG.
- **Ultralytics YOLOv8** : inférence de détection d’objets.
- **Vanilla JS / HTML / CSS** : contrôles et affichage du flux.

---

## 2. Pipeline de traitement vidéo

Chaque frame est traitée par `app/main.py` puis `app/vision.py`.

### Étapes du pipeline

1. Lecture de la source vidéo (`cv2.VideoCapture`).
2. Inference YOLOv8 sur la frame.
3. Séparation des détections en personnes et en équipements.
4. Association des équipements aux personnes.
5. Annotation de la frame avec des rectangles et du texte.
6. Encodage JPEG et émission via MJPEG.

### Processus

```
frame BGR
  │
  ▼
SafetyGearEngine.predict_and_annotate(frame)
  │
  ├─ Inférence YOLOv8
  ├─ Filtre classes personne / EPI
  ├─ Association EPI ↔ personne
  ├─ Annotation OpenCV
  └─ Retour d’une frame JPEG
```

### Format MJPEG

La route `/video_feed` renvoie un flux `multipart/x-mixed-replace` contenant
une suite d’images JPEG.

```text
Content-Type: multipart/x-mixed-replace; boundary=frame

--frame
Content-Type: image/jpeg

<bytes JPEG frame 1>
--frame
Content-Type: image/jpeg

<bytes JPEG frame 2>
...
```

Le navigateur interprète ces images comme une vidéo en continu.

---

## 3. Logique métier — association EPI ↔ personne

### Objectif

Associer chaque équipement détecté à la personne concernée et indiquer si
un équipement est manquant.

### Métrique d’association

Le calcul repose sur le chevauchement entre la bounding box de la personne
et celle de l’EPI.

```text
overlap_ratio = Aire(Intersection) / Aire(Personne)
```

### Avantage

Ce ratio est plus adapté que l’IoU pour les objets petits comme les casques.
Il mesure la proportion de la personne couverte par l’équipement.

### Groupes EPI

Exemple de groupes reconnus :

- **Casque** : `helmet`, `hardhat`, `headgear`, `hard-hat`
- **Gilet haute visibilité** : `vest`, `jacket`, `safety`, `hi-vis`, `hivis`

Si un groupe est absent pour une personne, la validation indique `MANQUANT`.

---

## 4. Flux d'utilisation

### 4.1 Webcam locale

1. Cliquer sur « Démarrer la caméra locale ».
2. Le backend ouvre `cv2.VideoCapture(0)`.
3. Le flux MJPEG est exposé sur `/video_feed`.
4. Le navigateur affiche la vidéo annotée.

### 4.2 Lecture de démonstration

1. Cliquer sur « Vidéo Démo ».
2. L’application utilise `assets/demo_chantier.mp4`.
3. La vidéo est relue en boucle.

### 4.3 Upload de vidéo

1. Envoyer un fichier MP4 via l’interface.
2. Le backend sauvegarde le fichier dans `static/uploads/`.
3. Le flux vidéo analysé est servi via `/video_feed`.

### 4.4 Upload de modèle

1. Envoyer un fichier `.pt` personnalisé.
2. Le backend enregistre le modèle dans `static/uploads/`.
3. Le modèle est rechargé à chaud pour l’inférence suivante.

---

## 5. Structure du projet

```text
/epi-surveillance
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── vision.py
├── assets/
│   └── demo_chantier.mp4
├── static/
│   ├── css/styles.css
│   ├── js/script.js
│   └── uploads/
├── templates/index.html
├── .github/workflows/python-app.yml
├── pyproject.toml
├── requirements.txt
├── README.md
└── PIPELINE.md
```

---

## 6. Lancement du serveur

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Démarrage

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Accès

| URL | Description |
|---|---|
| `http://localhost:8000/` | Dashboard principal |
| `http://localhost:8000/video_feed` | Flux MJPEG brut |
| `http://localhost:8000/docs` | API OpenAPI |
| `http://localhost:8000/redoc` | Documentation ReDoc |

---

## 7. Préparation GitHub

### GitHub Actions

Le workflow `.github/workflows/python-app.yml` vérifie :

- l’installation des dépendances
- la compilation Python des modules `app/main.py` et `app/vision.py`
- le packaging via `pyproject.toml`

### Fichiers locaux ignorés

- `.config/model_config.json`
- `static/uploads/`

### À vérifier avant push

- Fichiers volumineux (`.pt`, `.mp4`) sont-ils intentionnels ?
- Le README reflète-t-il bien les chemins et l’usage actuels ?
- Le pipeline CI doit rester simple et vérifiable.

---

> **SmartSafety Vision** — Système de surveillance EPI par IA
> Stack : FastAPI · Ultralytics YOLOv8 · OpenCV · Vanilla JS

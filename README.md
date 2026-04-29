# SmartSafety Vision — Surveillance EPI

Projet de démonstration de détection d'équipements de protection individuelle (EPI) en temps réel.
Le service utilise FastAPI, Ultralytics YOLOv8 et OpenCV pour analyser une webcam locale ou une vidéo uploadée.

## Fonctionnalités

- Détection de personnes et d'EPI en direct.
- Flux MJPEG pour afficher la vidéo annotée dans le navigateur.
- Upload de vidéos MP4 pour analyse frame par frame.
- Upload d'un modèle YOLOv8 personnalisé (`.pt`) pour des classes métier spécifiques.
- Dashboard Web simple avec contrôles et état d'alerte.

## Prérequis

- Python 3.10 ou plus.
- Accès à une webcam locale si vous utilisez le flux caméra.
- `git` pour cloner le dépôt.

## Installation

```bash
cd /home/mohamed/epi-surveillance
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> Optionnel : si vous utilisez `uv`, la commande `uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000` fonctionne aussi.

## Exécution

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Puis ouvrez :

```text
http://127.0.0.1:8000
```

## Utilisation

- **Démarrer la caméra locale** : lance le flux webcam.
- **Uploader une vidéo MP4** : analyse la vidéo et affiche les résultats.
- **Uploader un modèle YOLOv8 personnalisé** : charge un fichier `.pt` adapté à votre cas d'usage.

## Structure du projet

```text
/epi-surveillance
├── app
│   ├── __init__.py
│   ├── main.py
│   └── vision.py
├── assets
│   └── demo_chantier.mp4
├── static
│   ├── css
│   │   └── styles.css
│   ├── js
│   │   └── script.js
│   └── uploads
├── templates
│   └── index.html
├── .github
│   └── workflows
│       └── python-app.yml
├── pyproject.toml
├── requirements.txt
├── README.md
└── PIPELINE.md
```

## Notes importantes

- Le fichier `.config/model_config.json` est local et n'est pas suivi par Git.
- Les modèles `.pt` uploadés doivent rester dans `static/uploads/`.
- Le fichier `yolov8n.pt` en racine est un exemple de modèle de base.

## Préparation GitHub

- Le dépôt contient désormais un workflow GitHub Actions dans `.github/workflows/python-app.yml`.
- Le pipeline installe les dépendances et vérifie la compilation Python.
- Vérifiez avant `git commit` que les fichiers volumineux comme les modèles sont bien intentionnels.

## Auteur

SBII Mohamed

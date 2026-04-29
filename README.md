# Prototype de Surveillance de Sécurité EPI

Ce projet est un prototype local de surveillance de sécurité utilisant FastAPI, Ultralytics YOLOv8 et OpenCV.
Il analyse une webcam locale ou une vidéo MP4 uploadée, détecte les personnes et vérifie la présence d'équipements de protection individuelle (EPI).

## Arborescence du projet

```
/mehdi_tp_fr
├── app
│   ├── __init__.py
│   ├── main.py
│   └── vision.py
├── static
│   ├── css
│   │   └── styles.css
│   ├── js
│   │   └── script.js
│   └── uploads
├── templates
│   └── index.html
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Installation avec `uv`

1. Créez l'environnement virtuel :

```bash
cd /home/mohamed/mehdi_tp_fr
uv venv .venv
```

2. Installez les dépendances :

```bash
uv pip install -r requirements.txt
```

3. Lancez le serveur local :

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

4. Ouvrez votre navigateur sur :

```text
http://127.0.0.1:8000
```

## Utilisation

- `Démarrer la caméra locale` : lance le flux MJPEG depuis la webcam du serveur.
- `Uploader une vidéo MP4` : upload une vidéo et lance l'analyse frame par frame.
- `Charger un modèle YOLOv8 personnalisé` : téléversez votre `best.pt` spécialisé pour détecter casque, gilet, etc.

## Remarques

- Le backend utilise OpenCV pour capturer les frames et le modèle YOLOv8 pour l'inférence.
- Si vous souhaitez des alertes EPI précises, chargez un modèle personnalisé `best.pt` entraîné sur les classes de sécurité.
- En local, la caméra doit être accessible depuis le serveur, et le navigateur doit pouvoir afficher le flux MJPEG.

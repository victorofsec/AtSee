# Atsee hand capture

Petite app Python qui utilise OpenCV + MediaPipe pour detecter une main en webcam.
Quand le bout de l'index touche le bout du majeur pendant quelques frames, l'image
est gelee et l'app demande:

```text
Voulez-vous envoyer ?
```

Appuie sur `O`, `Y` ou `Entree` pour envoyer l'image a GitHub Models. Appuie sur
`N` ou `Echap` pour revenir a la camera. La reponse IA est affichee dans la
fenetre, dans le terminal, puis lue a voix haute avec une voix neuronale Edge TTS.
Pendant la lecture, appuie sur `P` pour mettre en pause, puis encore `P` pour
reprendre.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

La lecture vocale utilise `ffplay` ou `mpg123` si disponible sur la machine.

## Configuration

Le fichier `.env` peut contenir soit une cle brute seule, soit des variables:

```text
GITHUB_MODELS_TOKEN=github_pat_xxx
GITHUB_MODEL=openai/gpt-4.1
```

Variables optionnelles:

```text
CAMERA_INDEX=0
TOUCH_HOLD_FRAMES=5
MAX_IMAGE_SIDE=1024
JPEG_QUALITY=85
TTS_VOICE=fr-FR-DeniseNeural
TTS_RATE=+0%
HAND_LANDMARKER_MODEL_PATH=models/hand_landmarker.task
HAND_LANDMARKER_MODEL_URL=https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
GITHUB_MODELS_ENDPOINT=https://models.github.ai/inference/chat/completions
GITHUB_MODELS_API_VERSION=2026-03-10
```

Le token doit avoir l'autorisation `models:read`.

Au premier lancement, le modele MediaPipe officiel est telecharge dans
`models/hand_landmarker.task`.

## Source hand tracking

La boucle de detection suit l'approche OpenCV + MediaPipe du projet:
https://github.com/Sousannah/hand-tracking-using-mediapipe

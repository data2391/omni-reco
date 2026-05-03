# OMNI-RECO v2.1

> Moteur de reconnaissance faciale multi-composantes pour l'OSINT.  
> Fonctionne là où les autres abandonnent — profil, occlusion, basse qualité.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-green.svg)](https://fastapi.tiangolo.com/)

---

## Pourquoi OMNI-RECO ?

La plupart des outils (Dlib, InsightFace seul) échouent dès qu'un visage est de profil, partiellement masqué ou en basse résolution. OMNI-RECO combine **4 composantes biométriques indépendantes** avec une fusion pondérée adaptative. Si une composante est inutilisable (ex. géométrie sur profil dos), son poids est redistribué sur les autres. Le score reste fiable.

---

## Démo rapide

```bash
python compare.py photoA.jpg photoB.jpg --verbose
```

| Composante | Valeur | Détail |
|---|---|---|
| Score fusionné | 0.8801 | Méthode : arcface+texture+ear |
| ArcFace sim | 0.8453 | dist cosinus inverse |
| Texture Gabor | 0.9818 | 40 filtres zones peau |
| Oreilles YOLOv8 | 0.9160 | HOG 288D + Gabor 60D |

**MATCH** — Confiance : 88.0%

---

## Architecture

```
omni-reco/
├── core/
│   ├── faceengine.py          # Orchestrateur principal
│   ├── preprocessor_v2.py     # CLAHE + NlMeans + Sharpening, cap 1280px
│   ├── quality_scorer.py      # Score qualité image 0-1
│   ├── geometry_engine.py     # MediaPipe FaceMesh 468 pts, 18 ratios 3D
│   ├── texture_engine.py      # Gabor 40 filtres, zones peau
│   ├── ear_detector.py        # YOLOv8-pose, HOG 288D + Gabor 60D
│   ├── fusion_engine.py       # Fusion pondérée adaptative
│   └── regression_tests.py   # Suite de tests automatisés
├── api/
│   ├── main.py                # API REST FastAPI + Bearer token
│   ├── generate_token.py      # Générateur de tokens SHA-256
│   ├── launch-api.bat         # Démarrage Windows
│   └── launch-api.sh          # Démarrage Linux
├── compare.py                 # CLI
├── requirements_v2.txt        # 15 paquets versionnés
├── setup.bat                  # Installation Windows auto
└── setup-linux.sh             # Installation Linux toutes distros
```

### Composantes actives

| Composante | Modèle | Dimensions | Déclenchement |
|---|---|---|---|
| ArcFace | deep_buffalo_l / w600k_r50 | 512D | Toujours |
| Géométrie 3D | MediaPipe FaceMesh | 18 ratios, yaw<90° | yaw<90° |
| Texture Gabor | 40 filtres Gabor | 480D | Toujours |
| Oreilles | YOLOv8-pose | HOG 348D | yaw>15° |

---

## Installation

### Prérequis

- Python **3.10.x** obligatoire (MediaPipe 0.10.9 incompatible avec 3.11+)
- 4 Go RAM minimum
- 2 Go espace disque (modèles InsightFace + YOLOv8)

### Windows (automatique)

```bat
setup.bat
```

Détecte Python 3.10, le télécharge si absent, crée le venv, installe les 15 dépendances.

### Linux (toutes distributions)

```bash
chmod +x setup-linux.sh
./setup-linux.sh
```

Supporte automatiquement : Debian/Ubuntu/Mint · Fedora/RHEL/Rocky · Arch/Manjaro · openSUSE

---

## Utilisation

### CLI

```bash
# Activer l'environnement
launch.bat        # Windows
./launch.sh       # Linux

# Comparer deux photos
python compare.py photoA.jpg photoB.jpg --verbose

# Mode silencieux (sortie JSON)
python compare.py photoA.jpg photoB.jpg
```

### API REST

```bash
# Démarrage
api/launch-api.bat    # Windows
./api/launch-api.sh   # Linux
```

Génère automatiquement un token Bearer au 1er lancement.

#### Endpoints

| Méthode | URL | Auth | Description |
|---|---|---|---|
| GET | `/health` | Non | Statut moteur |
| POST | `/compare` | Bearer | Compare 2 photos |

#### Exemple

```bash
curl -X POST http://localhost:8000/compare \
  -H "Authorization: Bearer TONTOKEN" \
  -F "photoA=@photo1.jpg" \
  -F "photoB=@photo2.jpg"
```

#### Réponse

```json
{
  "match": true,
  "score": 0.8801,
  "confidence": 88.0,
  "method": "arcface+texture+ear",
  "detail": {
    "arcface_sim": 0.8453,
    "geo_score": null,
    "texture_score": 0.9818,
    "ear_score": 0.916
  },
  "severity": "high",
  "weights": { "arcface": 0.62, "texture": 0.12, "ear": 0.26 },
  "elapsed_s": 18.6
}
```

Documentation Swagger : `http://localhost:8000/docs`

---

## Interprétation des scores

| Score | Confiance | Signification |
|---|---|---|
| 0.90 – 1.00 | Très haute | Match quasi certain, conditions optimales |
| 0.80 – 0.89 | Haute | Match solide, angles/qualité complexes |
| 0.65 – 0.79 | Moyenne | Piste sérieuse, vérification recommandée |
| 0.50 – 0.64 | Faible | Incertain, qualité d'image insuffisante |
| < 0.50 | Nulle | Structures différentes |

---

## Correctifs v2.1 — 9 fixes

| Fix | Fichier | Description |
|---|---|---|
| FIX-1 | faceengine.py | Clé `quality_global`→`score_global` |
| FIX-2 | faceengine.py | InsightFace multi-tentatives padding +15 |
| FIX-3 | faceengine.py | Ranking sévérité inversé corrigé |
| FIX-4 | texture_engine.py | ROI Gabor 64×64 → 128×128 homogène |
| FIX-5 | fusion_engine.py | yaw≥90° → geo_score=None profil/dos |
| FIX-6 | setup.bat | mediapipe==0.10.9 --no-cache-dir auto-reinstall |
| FIX-7 | geometry_engine.py | Sanity-check yaw aberrant portrait serré |
| FIX-8 | ear_detector.py | PyTorch 2.6 compat add_safe_globals |
| FIX-9 | faceengine.py | Suppression FutureWarning InsightFace rcond |

---

## Dépendances clés

| Package | Version | Rôle |
|---|---|---|
| mediapipe | 0.10.9 | Géométrie 3D FaceMesh |
| insightface | 0.7.3 | ArcFace buffalo_l |
| onnxruntime | 1.17.3 | Inférence ONNX |
| ultralytics | 8.2.28 | YOLOv8-pose oreilles |
| fastapi | 0.111.0 | API REST |
| opencv-python | 4.9.0.80 | Vision + preprocessing |

> **Pourquoi mediapipe==0.10.9 exactement ?**  
> C'est la dernière version qui expose l'API `mp.solutions.face_mesh.FaceMesh` avec `refine_landmarks=True` nécessaire pour les 478 landmarks (iris inclus). Les versions ≥0.10.14 ont migré vers une API Tasks incompatible.

---

## GPU

Version actuelle : `CPUExecutionProvider` (ONNX Runtime). Pour activer GPU :

```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

InsightFace détectera automatiquement CUDA si disponible.

| Mode | Temps/comparaison |
|---|---|
| CPU | ~15-20s |
| GPU RTX 3060 | ~2-3s |

---

## Contribuer

Les Pull Requests sont ouvertes si tu comprends la différence entre une distance cosinus et une projection 3D.

1. Fork → branche `feature/ta-contribution`
2. Tests obligatoires : `python core/regression_tests.py`
3. Pas de modification de `requirements_v2.txt` sans justification documentée

---

## Avertissement légal

OMNI-RECO est un outil de recherche en OSINT et en biométrie, conçu pour les **audits de réputation numérique** et la **vérification d'identité dans un cadre légal**. L'utilisation de cet outil est soumise aux lois applicables dans votre juridiction, notamment le **RGPD** en Europe. L'utilisateur est seul responsable de l'usage qu'il en fait.

---

## Licence

MIT — voir [LICENSE](LICENSE)

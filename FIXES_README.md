# OMNI-RECO v2.1 — DÉFINITIF FINAL

## 9 correctifs intégrés

| Fix | Fichier | Description |
|-----|---------|-------------|
| FIX-1 | face_engine.py | Clé quality 'global_score' → 'global' |
| FIX-2 | face_engine.py | InsightFace multi-tentatives + padding 15% |
| FIX-3 | face_engine.py | Ranking sévérité inversé corrigé |
| FIX-4 | texture_engine.py | ROI Gabor 64×64 → 128×128 homogène |
| FIX-5 | fusion_engine.py | |yaw| > 90° → geo_score=None (profil/dos) |
| FIX-6 | setup.bat | mediapipe==0.10.9 --no-cache-dir + auto-reinstall |
| FIX-7 | geometry_engine.py | Sanity-check yaw aberrant (portrait serré) |
| FIX-8 | ear_detector.py | PyTorch >= 2.6 weights_only compat + safe_globals |
| FIX-9 | face_engine.py | Suppression FutureWarning InsightFace rcond |

## Structure
```
core/                     moteur OMNI-RECO (9 fixes)
api/
  main.py                 serveur FastAPI
  generate_token.py       token Bearer
  launch-api.bat          lance l'API Windows
  test-api.bat            teste l'API avec curl
  README.md
compare.py                CLI
main.py
requirements_v2.txt       15 paquets versionnés
setup.bat                 installation Windows auto
setup-linux.sh            installation Linux toutes distros
```

## Composantes actives (toutes conditions)
- ArcFace (buffalo_l) — embedding 512D
- Géométrie 3D MediaPipe — 18 ratios, corrigé yaw
- Texture Gabor — 40 filtres × zones peau
- Oreilles YOLOv8-pose — HOG 288D + Gabor 60D

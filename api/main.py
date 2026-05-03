"""
api/main.py — OMNI-RECO API v1.0
FastAPI REST — comparaison faciale via token Bearer

Endpoints :
  POST /compare        → compare deux images (form-data)
  GET  /health         → statut moteur

Auth : Bearer token dans header Authorization
  Authorization: Bearer <TOKEN>

Démarrage :
  cd api
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Générer un token :
  python generate_token.py
"""

import sys, os, io, time, secrets, hashlib
from pathlib import Path
from typing import Optional

# Ajoute le dossier parent (racine OMNI-RECO) au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

from core.face_engine import FaceEngine

# ── Config ────────────────────────────────────────────────────────────────────

TOKENS_FILE = Path(__file__).parent / "tokens.txt"

def _load_tokens() -> set:
    """Charge les tokens valides depuis tokens.txt (un token hashé par ligne)."""
    if not TOKENS_FILE.exists():
        return set()
    return set(l.strip() for l in TOKENS_FILE.read_text().splitlines() if l.strip())

def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()

# ── Sécurité ──────────────────────────────────────────────────────────────────

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token_hash = _token_hash(credentials.credentials)
    if token_hash not in _load_tokens():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide ou expiré",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="OMNI-RECO API",
    description="Reconnaissance faciale multi-composantes (ArcFace + Géométrie + Texture + Oreilles)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

_engine: Optional[FaceEngine] = None

def get_engine() -> FaceEngine:
    global _engine
    if _engine is None:
        _engine = FaceEngine()
    return _engine

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Statut du moteur — pas d'authentification requise."""
    engine = get_engine()
    return {
        "status": "ok",
        "engine": "OMNI-RECO v2.1",
        "insightface": True,
    }

@app.post("/compare")
async def compare_faces(
    photo_a: UploadFile = File(..., description="Image A (jpg/png)"),
    photo_b: UploadFile = File(..., description="Image B (jpg/png)"),
    token: str = Depends(verify_token),
):
    """
    Compare deux photos de visage.

    - **photo_a** : première image (multipart/form-data)
    - **photo_b** : deuxième image (multipart/form-data)

    Retourne le score de similarité, le verdict MATCH/NO MATCH,
    et le détail de chaque composante (ArcFace, Géométrie, Texture, Oreilles).
    """
    t0 = time.time()
    engine = get_engine()

    try:
        bytes_a = await photo_a.read()
        bytes_b = await photo_b.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lecture image échouée : {e}")

    if not bytes_a or not bytes_b:
        raise HTTPException(status_code=400, detail="Images vides")

    try:
        result = engine.process_pair_sync(io.BytesIO(bytes_a), io.BytesIO(bytes_b))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur moteur : {e}")

    elapsed = round(time.time() - t0, 2)

    if result.get("method") == "error" or result.get("error"):
        return {
            "match": False,
            "score": 0.0,
            "confidence": 0.0,
            "method": "error",
            "error": result.get("error", "Visage non détecté"),
            "elapsed_s": elapsed,
        }

    return {
        "match":      result["match"],
        "score":      result["score"],
        "confidence": result["confidence"],
        "method":     result["method"],
        "detail": {
            "arcface_sim":    result["detail"].get("arcface_sim"),
            "geo_score":      result["detail"].get("geo_score"),
            "texture_score":  result["detail"].get("texture_score"),
            "ear_score":      result["detail"].get("ear_score"),
            "severity":       result["detail"].get("severity"),
            "partial_a":      result["detail"].get("partial_a"),
            "partial_b":      result["detail"].get("partial_b"),
            "yaw_a":          result["detail"].get("yaw_a"),
            "yaw_b":          result["detail"].get("yaw_b"),
            "weights":        result["detail"].get("weights"),
        },
        "elapsed_s": elapsed,
    }

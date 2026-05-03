"""
core/face_engine.py  —  OMNI-RECO v2.0
Chef d'orchestre — Reconnaissance faciale complète

Pipeline process_single() :
  1. preprocessor_v2  → image nettoyée
  2. InsightFace       → embedding ArcFace 512D
  3. geometry_engine   → signature géométrique 3D (18 ratios)
  4. texture_engine    → signature texture Gabor (320D)
  5. quality_scorer    → score qualité + sévérité
  6. fusion_engine     → verdict final

Gestion native :
  - Visages partiels (yaw > 25°)
  - Images petites (< 128px → ESRGAN automatique)
  - Fallback ArcFace-only si MediaPipe échoue
  - Threading async (run_in_executor, ne bloque pas l'event loop)

Usage rapide :
  engine = FaceEngine()
  result = await engine.process_pair(img_a, img_b)
  # → {"match": True, "score": 0.81, "confidence": 81.0, ...}
"""

import io
import asyncio
import functools
import traceback
from pathlib import Path
from typing import Union, Optional

import cv2
import warnings
import numpy as np
# [FIX-9] Supprimer FutureWarning InsightFace rcond (bénin, hors de notre contrôle)
warnings.filterwarnings(
    "ignore",
    message=".*rcond.*parameter will change.*",
    category=FutureWarning,
    module="insightface.*"
)


# ── Modules OMNI-RECO ─────────────────────────────────────────────────────────
from core.preprocessor_v2 import preprocess
from core.geometry_engine  import GeometryEngine
from core.texture_engine   import get_texture_engine
from core.quality_scorer   import score_image_quality as score_image
from core.fusion_engine    import fuse, arcface_similarity

# ── Type alias ────────────────────────────────────────────────────────────────
ImageInput = Union[str, Path, bytes, io.BytesIO]


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

# Si le visage crop est < ce seuil (en px entre les yeux), on signale "partial"
MIN_IPD_PX = 30

# Modèle InsightFace par défaut
DEFAULT_MODEL = "buffalo_l"

# Seuil yaw au-delà duquel on active le mode visage partiel
YAW_PARTIAL_THRESHOLD = 25.0


# ══════════════════════════════════════════════════════════════════════════════
#  CHARGEMENT INSIGHTFACE (lazy, une fois par session)
# ══════════════════════════════════════════════════════════════════════════════

_INSIGHT_APP   = None
_INSIGHT_LOCK  = asyncio.Lock()


async def _get_insight_app(model_name: str = DEFAULT_MODEL):
    global _INSIGHT_APP
    if _INSIGHT_APP is not None:
        return _INSIGHT_APP

    async with _INSIGHT_LOCK:
        if _INSIGHT_APP is not None:
            return _INSIGHT_APP
        loop = asyncio.get_event_loop()
        _INSIGHT_APP = await loop.run_in_executor(
            None, _load_insight_sync, model_name
        )
    return _INSIGHT_APP


def _load_insight_sync(model_name: str):
    """Chargement synchrone InsightFace (appel dans executor)."""
    try:
        import insightface
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app
    except Exception as e:
        raise RuntimeError(f"[FaceEngine] Impossible de charger InsightFace ({model_name}): {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _bytesio_to_bgr(bio: io.BytesIO) -> Optional[np.ndarray]:
    bio.seek(0)
    arr = np.frombuffer(bio.read(), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _to_bytesio(image: ImageInput) -> io.BytesIO:
    if isinstance(image, io.BytesIO):
        image.seek(0)
        return image
    if isinstance(image, (str, Path)):
        with open(str(image), "rb") as f:
            return io.BytesIO(f.read())
    return io.BytesIO(image if isinstance(image, bytes) else bytes(image))


def _get_embedding_sync(app, img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Extrait l'embedding ArcFace depuis un array BGR (synchrone).

    [FIX-2] Stratégie multi-tentatives :
    1. Resize à 640px (optimal InsightFace det_size=640)
    2. Si échec → ajouter 15% padding (portrait serré qui déborde du cadre)
    3. Si échec → essai à 480px (visage très grand dans le frame)
    4. Si échec → essai image originale telle quelle
    """
    def _best_face(faces):
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))

    def _resize_to(img, max_side):
        h, w = img.shape[:2]
        if max(h, w) <= max_side:
            return img
        scale = max_side / max(h, w)
        return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    def _add_padding(img, ratio=0.15):
        h, w = img.shape[:2]
        pad_h = int(h * ratio)
        pad_w = int(w * ratio)
        return cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)

    # Tentative 1 : resize optimal 640px
    img_640 = _resize_to(img_bgr, 640)
    face = _best_face(app.get(img_640))
    if face is not None:
        return face.embedding.astype(np.float32)

    # Tentative 2 : padding 15% + resize 640px (portrait serré / visage tronqué)
    img_pad = _add_padding(img_bgr, 0.15)
    img_pad_640 = _resize_to(img_pad, 640)
    face = _best_face(app.get(img_pad_640))
    if face is not None:
        return face.embedding.astype(np.float32)

    # Tentative 3 : resize 480px (visage trop grand dans le frame)
    img_480 = _resize_to(img_bgr, 480)
    face = _best_face(app.get(img_480))
    if face is not None:
        return face.embedding.astype(np.float32)

    # Tentative 4 : image originale sans resize
    face = _best_face(app.get(img_bgr))
    if face is not None:
        return face.embedding.astype(np.float32)

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  PROCESS SINGLE — cœur du moteur
# ══════════════════════════════════════════════════════════════════════════════

async def _process_single_async(
    image:      ImageInput,
    log_fn=None,
    model_name: str = DEFAULT_MODEL,
) -> dict:
    """
    Traite une image → génère sa fiche d'identité biométrique complète.

    Retourne :
    {
        "ok": True,
        "embedding": np.ndarray (512D),
        "geo_sig":   dict (geometry_engine output),
        "tex_sig":   dict (texture_engine output),
        "quality":   dict (quality_scorer output),
        "is_partial": bool,
        "yaw": float,
        "log": [str]
    }
    """
    logs = []

    def _log(msg: str):
        logs.append(msg)
        if log_fn:
            try:
                log_fn(msg)
            except Exception:
                pass

    try:
        bio = _to_bytesio(image)

        # ── Étape 1 : Prétraitement ──────────────────────────────────────────
        _log("[FaceEngine] Étape 1/5 : Prétraitement")
        loop = asyncio.get_event_loop()
        bio_clean = await loop.run_in_executor(
            None, functools.partial(preprocess, bio, _log)
        )
        img_bgr = _bytesio_to_bgr(bio_clean)
        if img_bgr is None:
            return {"ok": False, "reason": "Décodage image échoué", "log": logs}

        # ── Étape 2 : Qualité image ──────────────────────────────────────────
        _log("[FaceEngine] Étape 2/5 : Score qualité")
        bio_clean.seek(0)
        quality = await loop.run_in_executor(
            None, score_image, bio_clean
        )
        severity = quality.get("severity", "medium")  # [FIX-1] clé corrigée
        _log(f"[FaceEngine] Qualité : {quality.get('global', 0):.2f} | sévérité : {severity}")

        # ── Étape 3 : ArcFace embedding ──────────────────────────────────────
        _log("[FaceEngine] Étape 3/5 : ArcFace embedding")
        app = await _get_insight_app(model_name)
        embedding = await loop.run_in_executor(
            None, _get_embedding_sync, app, img_bgr
        )
        if embedding is None:
            _log("[FaceEngine] ⚠ Aucun visage détecté par InsightFace")
            return {"ok": False, "reason": "Aucun visage détecté", "log": logs}

        # ── Étape 4 : Géométrie MediaPipe ────────────────────────────────────
        _log("[FaceEngine] Étape 4/5 : Géométrie 3D MediaPipe")
        geo_engine = GeometryEngine()
        if not geo_engine.is_ready:
            _log("[FaceEngine] ⚠  MediaPipe non disponible — Géométrie désactivée")
            _log("[FaceEngine]    → pip install mediapipe==0.10.9  (Python 3.10)")
            geo_sig = {"ok": False, "reason": "mediapipe_unavailable"}
        else:
            bio_clean.seek(0)
            geo_sig = await loop.run_in_executor(
                None, geo_engine.extract, bio_clean
            )

        yaw        = geo_sig.get("pose", {}).get("yaw", 0.0) if geo_sig.get("ok") else 0.0
        is_partial = abs(yaw) > YAW_PARTIAL_THRESHOLD or not geo_sig.get("ok", False)
        if is_partial:
            _log(f"[FaceEngine] ⚠ Visage partiel détecté (yaw={yaw:.1f}°)")

        # ── Étape 5 : Texture Gabor ──────────────────────────────────────────
        _log("[FaceEngine] Étape 5/5 : Texture Gabor")
        tex_engine = get_texture_engine()
        landmarks  = None
        if geo_sig.get("ok") and "landmarks_px" in geo_sig:
            landmarks = geo_sig["landmarks_px"]

        bio_clean.seek(0)
        tex_sig = await loop.run_in_executor(
            None, functools.partial(tex_engine.extract, bio_clean, landmarks)
        )
        _log(f"[FaceEngine] Texture : méthode={tex_sig.get('method')} "
             f"| zones={tex_sig.get('zones_used', [])}")

        _log("[FaceEngine] ✓ Fiche biométrique générée")
        return {
            "ok":         True,
            "embedding":  embedding,
            "geo_sig":    geo_sig,
            "tex_sig":    tex_sig,
            "quality":    quality,
            "severity":   severity,
            "is_partial": is_partial,
            "yaw":        yaw,
            "log":        logs,
        }

    except Exception as e:
        _log(f"[FaceEngine] ERREUR : {e}")
        _log(traceback.format_exc())
        return {"ok": False, "reason": str(e), "log": logs}


# ══════════════════════════════════════════════════════════════════════════════
#  FACE ENGINE — Interface publique
# ══════════════════════════════════════════════════════════════════════════════

class FaceEngine:
    """
    Interface principale OMNI-RECO v2.0.

    Usage async :
        engine = FaceEngine()
        pair   = await engine.process_pair(img_a, img_b)

    Usage sync (pour scripts CLI) :
        pair = engine.process_pair_sync(img_a, img_b)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, log_fn=None):
        self.model_name = model_name
        self.log_fn     = log_fn

    # ── API ASYNC ─────────────────────────────────────────────────────────────

    async def process_single(self, image: ImageInput) -> dict:
        """Génère la fiche biométrique complète d'une image."""
        return await _process_single_async(image, self.log_fn, self.model_name)

    async def process_pair(
        self,
        image_a: ImageInput,
        image_b: ImageInput,
    ) -> dict:
        """
        Compare deux images faciales.

        Traitement parallèle des deux images (asyncio.gather).

        Retourne :
        {
            "match":      bool,
            "score":      float [0-1],
            "confidence": float [0-100%],
            "method":     str,
            "detail": {
                "arcface_sim":   float,
                "geo_score":     float | None,
                "texture_score": float | None,
                "severity":      str,
                "partial_a":     bool,
                "partial_b":     bool,
            },
            "log_a": [str],
            "log_b": [str],
        }
        """
        # Traitement parallèle
        fiche_a, fiche_b = await asyncio.gather(
            self.process_single(image_a),
            self.process_single(image_b),
        )

        # Échec sur l'une des deux images
        if not fiche_a.get("ok"):
            return {"match": False, "score": 0.0, "confidence": 0.0,
                    "error": f"Image A : {fiche_a.get('reason')}", **_empty_detail()}
        if not fiche_b.get("ok"):
            return {"match": False, "score": 0.0, "confidence": 0.0,
                    "error": f"Image B : {fiche_b.get('reason')}", **_empty_detail()}

        # ── Score ArcFace ────────────────────────────────────────────────────
        arcface_sim = arcface_similarity(fiche_a["embedding"], fiche_b["embedding"])

        # ── Score Géométrie (si les deux ont réussi) ─────────────────────────
        geo_score = None
        if fiche_a["geo_sig"].get("ok") and fiche_b["geo_sig"].get("ok"):
            geo_engine = GeometryEngine()
        if not geo_engine.is_ready:
            if log_fn: log_fn("[FaceEngine] ⚠  MediaPipe non disponible — Géométrie désactivée")
            if log_fn: log_fn("[FaceEngine]    → pip install mediapipe==0.10.9  (Python ≤3.11)")
            result["geo_sig"] = {"ok": False, "reason": "mediapipe_unavailable"}
        else:
            geo_result = geo_engine.compare(fiche_a["geo_sig"], fiche_b["geo_sig"])
            geo_score  = geo_result.get("score")

        # ── Score Texture ────────────────────────────────────────────────────
        texture_score = None
        if fiche_a["tex_sig"].get("ok") and fiche_b["tex_sig"].get("ok"):
            tex_result    = get_texture_engine().compare(fiche_a["tex_sig"], fiche_b["tex_sig"])
            texture_score = tex_result.get("score")

        # ── Sévérité : prend la pire des deux ───────────────────────────────
        sev_rank = {"high": 2, "medium": 1, "low": 0}
        sev_a    = fiche_a.get("severity", "medium")
        sev_b    = fiche_b.get("severity", "medium")
        severity = sev_a if sev_rank[sev_a] <= sev_rank[sev_b] else sev_b

        # ── Fusion finale ────────────────────────────────────────────────────
        yaw_max_v = max(abs(fiche_a.get("yaw", 0.0)), abs(fiche_b.get("yaw", 0.0)))
        verdict = fuse(arcface_sim, geo_score, texture_score, severity, yaw_max=yaw_max_v)

        return {
            "match":      verdict["match"],
            "score":      verdict["fused_score"],
            "confidence": verdict["confidence"],
            "method":     verdict["method"],
            "detail": {
                "arcface_sim":   arcface_sim,
                "geo_score":     geo_score,
                "texture_score": texture_score,
                "severity":      severity,
                "partial_a":     fiche_a.get("is_partial", False),
                "partial_b":     fiche_b.get("is_partial", False),
                "yaw_a":         fiche_a.get("yaw", 0.0),
                "yaw_b":         fiche_b.get("yaw", 0.0),
                "weights":       verdict["detail"]["weights_used"],
            },
            "log_a": fiche_a.get("log", []),
            "log_b": fiche_b.get("log", []),
        }

    # ── API SYNC (CLI / scripts) ──────────────────────────────────────────────

    def process_pair_sync(self, image_a: ImageInput, image_b: ImageInput) -> dict:
        """Version synchrone de process_pair pour usage CLI."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.process_pair(image_a, image_b))
                    return future.result()
            return loop.run_until_complete(self.process_pair(image_a, image_b))
        except RuntimeError:
            return asyncio.run(self.process_pair(image_a, image_b))

    def process_single_sync(self, image: ImageInput) -> dict:
        """Version synchrone de process_single."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.process_single(image))
        except RuntimeError:
            return asyncio.run(self.process_single(image))


def _empty_detail() -> dict:
    return {"detail": {
        "arcface_sim": None, "geo_score": None, "texture_score": None,
        "severity": "low", "partial_a": False, "partial_b": False,
    }}


# ══════════════════════════════════════════════════════════════════════════════
#  PATCH v2.1 — Intégration EarDetector dans process_single
# ══════════════════════════════════════════════════════════════════════════════
# Ce patch ajoute l'étape 6 : extraction auriculaire via EarDetector
# Activée automatiquement si yaw > 15° ou si landmarks disponibles

from core.ear_detector import get_ear_detector

async def _process_single_v21(
    image:      ImageInput,
    log_fn=None,
    model_name: str = DEFAULT_MODEL,
) -> dict:
    """
    v2.1 : identique à _process_single_async + étape 6 oreilles.
    Remplace process_single() dans FaceEngine quand ear_detector est disponible.
    """
    result = await _process_single_async(image, log_fn, model_name)
    if not result.get("ok"):
        return result

    # ── Étape 6 : Oreilles (si yaw > 15° ou mode full) ──────────────────────
    yaw         = result.get("yaw", 0.0)
    landmarks   = result.get("geo_sig", {}).get("landmarks_px")
    ear_det     = get_ear_detector()
    if log_fn:
        ear_det.set_log_fn(log_fn)

    loop     = asyncio.get_event_loop()
    # Recharge l'image source (on n'a plus le BytesIO original ici)
    ear_sigs = []
    if abs(yaw) > 15.0 or landmarks is not None:
        # On reprocesse depuis la fiche (image non disponible ici → skip si absente)
        result["log"].append(
            f"[FaceEngine] Étape 6 : oreilles (yaw={yaw:.1f}°) — intégré en amont"
        )
    result["ear_sigs"] = ear_sigs
    return result


# Monkey-patch sur FaceEngine pour activer la v2.1
_original_process_single = FaceEngine.process_single

async def _patched_process_single(self, image: ImageInput) -> dict:
    result = await _process_single_async(image, self.log_fn, self.model_name)
    if not result.get("ok"):
        return result
    # Extraction auriculaire
    yaw       = result.get("yaw", 0.0)
    landmarks = result.get("geo_sig", {}).get("landmarks_px")
    ear_det   = get_ear_detector()
    if self.log_fn:
        ear_det.set_log_fn(self.log_fn)
    loop = asyncio.get_event_loop()
    ear_sigs = await loop.run_in_executor(
        None,
        lambda: ear_det.extract(image, landmarks_pts=landmarks, yaw=yaw)
    )
    result["ear_sigs"] = ear_sigs
    return result

FaceEngine.process_single = _patched_process_single


# ══════════════════════════════════════════════════════════════════════════════
#  PATCH v2.1b — process_pair intègre ear_score dans fusion
# ══════════════════════════════════════════════════════════════════════════════

async def _patched_process_pair(self, image_a: ImageInput, image_b: ImageInput) -> dict:
    """
    Étend process_pair() : extrait les signatures auriculaires et les passe
    à fusion_engine.fuse() comme 4ème composante.
    """
    from core.ear_detector import get_ear_detector
    from core.fusion_engine import fuse, arcface_similarity

    # Traitement parallèle des deux images
    fiche_a, fiche_b = await asyncio.gather(
        _process_single_async(image_a, self.log_fn, self.model_name),
        _process_single_async(image_b, self.log_fn, self.model_name),
    )

    # Vérification minimale
    if not fiche_a.get("ok") or not fiche_b.get("ok"):
        err = fiche_a.get("error") or fiche_b.get("error") or "Visage non détecté"
        return {"match": False, "score": 0.0, "confidence": 0.0,
                "method": "error", "detail": {}, "error": err}

    # ── ArcFace ──────────────────────────────────────────────────────────────
    emb_a = fiche_a["embedding"]
    emb_b = fiche_b["embedding"]
    arc_sim = arcface_similarity(emb_a, emb_b)

    # ── Géométrie ────────────────────────────────────────────────────────────
    geo_score = None
    if fiche_a.get("geo_sig", {}).get("ok") and fiche_b.get("geo_sig", {}).get("ok"):
        geo_result = GeometryEngine().compare(fiche_a["geo_sig"], fiche_b["geo_sig"])
        geo_score  = geo_result.get("score")

    # ── Texture ──────────────────────────────────────────────────────────────
    tex_score = None
    if fiche_a.get("tex_sig", {}).get("ok") and fiche_b.get("tex_sig", {}).get("ok"):
        tex_result = get_texture_engine().compare(fiche_a["tex_sig"], fiche_b["tex_sig"])
        tex_score  = tex_result.get("score")

    # ── Oreilles ─────────────────────────────────────────────────────────────
    ear_score  = None
    ear_detail = {}
    yaw_a = fiche_a.get("yaw", 0.0)
    yaw_b = fiche_b.get("yaw", 0.0)
    if abs(yaw_a) > 15.0 or abs(yaw_b) > 15.0:
        ear_det = get_ear_detector()
        if self.log_fn:
            ear_det.set_log_fn(self.log_fn)
        loop = asyncio.get_event_loop()
        lm_a = fiche_a.get("geo_sig", {}).get("landmarks_px")
        lm_b = fiche_b.get("geo_sig", {}).get("landmarks_px")
        sigs_a = await loop.run_in_executor(
            None, lambda: ear_det.extract(image_a, landmarks_pts=lm_a, yaw=yaw_a))
        sigs_b = await loop.run_in_executor(
            None, lambda: ear_det.extract(image_b, landmarks_pts=lm_b, yaw=yaw_b))
        ear_result = ear_det.compare(sigs_a, sigs_b)
        if ear_result.get("ok"):
            ear_score  = ear_result["score"]
            ear_detail = ear_result.get("detail", {})
            if self.log_fn:
                self.log_fn(f"[FaceEngine] ear_score={ear_score:.4f}")

    # ── Sévérité (pire des deux images) [FIX-3] ─────────────────────────────
    # low=0 (pire qualité), high=2 (meilleure) — on prend le rang le plus bas
    sev_rank = {"high": 2, "medium": 1, "low": 0}
    sev_a = fiche_a.get("severity", "medium")
    sev_b = fiche_b.get("severity", "medium")
    severity = sev_a if sev_rank.get(sev_a, 1) <= sev_rank.get(sev_b, 1) else sev_b

    is_partial = fiche_a.get("is_partial", False) or fiche_b.get("is_partial", False)

    # ── Fusion 4 composantes ─────────────────────────────────────────────────
    yaw_max_val = max(abs(yaw_a), abs(yaw_b))
    fusion = fuse(
        arcface_sim      = arc_sim,
        geo_score        = geo_score,
        texture_score    = tex_score,
        quality_severity = severity,
        ear_score        = ear_score,
        is_partial       = is_partial,
        yaw_max          = yaw_max_val,
    )

    # ── Construction réponse finale ───────────────────────────────────────────
    detail = {
        **fusion["detail"],
        "arcface_sim":   arc_sim,
        "geo_score":     geo_score,
        "texture_score": tex_score,
        "ear_score":     ear_score,
        "ear_detail":    ear_detail,
        "severity":      severity,
        "partial_a":     fiche_a.get("is_partial", False),
        "partial_b":     fiche_b.get("is_partial", False),
        "yaw_a":         round(yaw_a, 1),
        "yaw_b":         round(yaw_b, 1),
        "weights":       fusion["detail"]["weights_used"],
    }

    return {
        "match":      fusion["match"],
        "score":      fusion["fused_score"],
        "confidence": fusion["confidence"],
        "method":     fusion["method"],
        "threshold":  fusion["threshold"],
        "detail":     detail,
    }


# Monkey-patch final
FaceEngine.process_pair = _patched_process_pair

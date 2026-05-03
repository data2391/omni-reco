"""
core/quality_scorer.py  —  OMNI-RECO v2.0
Quality Intelligence — BRISQUE + Occlusion + Blur + Brightness
Note globale [0.0 - 1.0] pour piloter la sévérité adaptative du fusion_engine.
"""

import io
import cv2
import numpy as np
from pathlib import Path
from typing import Union


def _load_bgr(image: Union[str, Path, bytes, io.BytesIO]):
    if isinstance(image, (str, Path)):
        return cv2.imread(str(image))
    raw = image.read() if hasattr(image, "read") else image
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ── 1. Blur (Laplacien) ───────────────────────────────────────────────────────
def _blur_score(gray: np.ndarray) -> float:
    """Score 0-1 : 1 = net, 0 = flou. Seuil calibré empiriquement."""
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # 500 = image très nette, 30 = flou prononcé
    return float(min(1.0, lap_var / 500.0))


# ── 2. Brightness / Exposition ────────────────────────────────────────────────
def _brightness_score(gray: np.ndarray) -> float:
    """Pénalise sur-/sous-exposition. Optimal = mean luminance 100-160."""
    mean = float(gray.mean())
    if 80 <= mean <= 180:
        return 1.0
    elif mean < 80:
        return max(0.0, mean / 80.0)
    else:
        return max(0.0, 1.0 - (mean - 180) / 75.0)


# ── 3. Contraste (écart-type) ─────────────────────────────────────────────────
def _contrast_score(gray: np.ndarray) -> float:
    std = float(gray.std())
    return float(min(1.0, std / 60.0))


# ── 4. BRISQUE simplifié (NSS features) ──────────────────────────────────────
def _brisque_lite(gray: np.ndarray) -> float:
    """
    Version allégée du BRISQUE.
    Calcule les statistiques MSCN (Mean Subtracted Contrast Normalized).
    Retourne un score 0-1 (1 = bonne qualité, 0 = artefacts).
    """
    gray_f = gray.astype(np.float64)
    blur   = cv2.GaussianBlur(gray_f, (7, 7), 7.0 / 6.0)
    blur_sq = cv2.GaussianBlur(gray_f**2, (7, 7), 7.0 / 6.0)
    sigma  = np.sqrt(np.maximum(0, blur_sq - blur**2))
    # MSCN coefficients
    mscn   = (gray_f - blur) / (sigma + 1.0)
    # Paramètres GGD : kurtosis mesure les artefacts de compression/bruit
    kurt   = float(cv2.mean(mscn**4)[0])
    # Image naturelle → kurtosis ~3. Très compressée ou bruitée → > 8
    score  = max(0.0, 1.0 - max(0.0, (kurt - 3.0) / 15.0))
    return round(score, 4)


# ── 5. Occlusion estimée (LBP facial) ────────────────────────────────────────
def _occlusion_score(gray: np.ndarray, face_bbox=None) -> float:
    """
    Détecte les zones uniformes (masque, main, flou partiel) via la variance locale.
    Découpe l'image en 4x4 blocs, pénalise les blocs à très faible variance.
    Retourne 1.0 si aucune occlusion probable, < 0.5 si > 30% de la surface bloquée.
    """
    if face_bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in face_bbox]
        roi = gray[y1:y2, x1:x2]
    else:
        roi = gray

    if roi.size == 0:
        return 0.5

    roi_r = cv2.resize(roi, (64, 64))
    h, w  = roi_r.shape
    rows, cols = 4, 4
    bh, bw = h // rows, w // cols
    low_var_blocks = 0

    for r in range(rows):
        for c in range(cols):
            block = roi_r[r*bh:(r+1)*bh, c*bw:(c+1)*bw]
            if block.std() < 8.0:   # bloc uniforme = zone cachée
                low_var_blocks += 1

    occ_ratio = low_var_blocks / (rows * cols)
    return round(max(0.0, 1.0 - occ_ratio * 1.5), 4)


# ══════════════════════════════════════════════════════════════════════════════
#  API PUBLIQUE
# ══════════════════════════════════════════════════════════════════════════════

def score_image_quality(
    image: Union[str, Path, bytes, io.BytesIO],
    face_bbox=None,
) -> dict:
    """
    Retourne un dict complet de qualité image.
    face_bbox : [x1, y1, x2, y2] optionnel pour centrer l'analyse sur le visage.

    Scores retournés :
      "blur"        : netteté [0-1]
      "brightness"  : exposition [0-1]
      "contrast"    : contraste [0-1]
      "brisque"     : qualité globale sans référence [0-1]
      "occlusion"   : absence d'occlusion [0-1]
      "global"      : score global pondéré [0-1]
      "severity"    : "high" / "medium" / "low" pour piloter le threshold ArcFace
    """
    img_bgr = _load_bgr(image)
    if img_bgr is None:
        return {"ok": False, "global": 0.0, "severity": "low"}

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    blur       = _blur_score(gray)
    brightness = _brightness_score(gray)
    contrast   = _contrast_score(gray)
    brisque    = _brisque_lite(gray)
    occlusion  = _occlusion_score(gray, face_bbox)

    # Pondération : netteté et occlusion sont les plus critiques pour ArcFace
    global_score = round(
        0.30 * blur +
        0.20 * brightness +
        0.10 * contrast +
        0.20 * brisque +
        0.20 * occlusion,
        4
    )

    # Sévérité adaptative → pilote le threshold ArcFace dans fusion_engine
    if global_score >= 0.75:
        severity = "high"    # seuil strict (0.40)
    elif global_score >= 0.45:
        severity = "medium"  # seuil normal (0.45)
    else:
        severity = "low"     # seuil souple (0.52) + avertissement

    return {
        "ok":         True,
        "scores": {
            "blur":       blur,
            "brightness": brightness,
            "contrast":   contrast,
            "brisque":    brisque,
            "occlusion":  occlusion,
        },
        "global":   global_score,
        "severity": severity,
    }


# Seuils ArcFace adaptés à la sévérité qualité image
ADAPTIVE_THRESHOLD = {
    "high":   0.40,
    "medium": 0.45,
    "low":    0.52,
}

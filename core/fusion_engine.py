"""
core/fusion_engine.py  —  OMNI-RECO v2.1
Fusion Score = ArcFace + Géométrie 3D + Texture Gabor + Oreilles (HOG+Gabor)

4 composantes — pondération TOTALEMENT adaptative :
  - Chaque composante manquante (None) est exclue, les poids se redistribuent
  - Les oreilles ont un poids BONUS si le visage est partiel (yaw > 25°)
  - La texture a un poids pénalisé si severity=low (sensible à l'éclairage)
  - Un score ear disponible sur portrait 3/4 peut compenser l'ArcFace dégradé

Scores en entrée (tous en similarité [0-1], pas en distance) :
  arcface_sim  : 1 - arcface_distance            (face_engine)
  geo_score    : geometry_engine.compare()["score"]
  texture_score: texture_engine.compare()["score"]
  ear_score    : ear_detector.compare()["score"]

Score de sortie : fused [0-1], 1 = identique.
"""

import numpy as np
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
#  TABLE DE PONDÉRATION DE BASE  (4 composantes)
# ══════════════════════════════════════════════════════════════════════════════
#  Lecture : pour chaque niveau de sévérité, poids de base par composante.
#  Ces poids sont ensuite modifiés dynamiquement selon les données disponibles.

_BASE_WEIGHTS = {
    # sév.     ArcFace  Géo    Texture  Oreilles
    "high":   (0.60,   0.22,   0.09,   0.09),
    "medium": (0.50,   0.27,   0.13,   0.10),
    "low":    (0.35,   0.35,   0.15,   0.15),
}

# Seuils de match (similarité)
THRESHOLDS_SIM = {
    "high":   0.60,
    "medium": 0.55,
    "low":    0.48,
}

# Bonus poids oreilles si visage partiel sur AU MOINS UNE des deux images
EAR_PARTIAL_BONUS = 0.12    # ajout au poids oreille, soustrait à arcface
GEO_PARTIAL_BONUS = 0.08    # ajout géométrie si partial (invariant à rotation)

# Malus texture si sévérité low (très sensible aux conditions lumineuses)
TEX_LOW_MALUS = 0.05


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def arcface_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Distance cosinus normalisée → similarité [0-1]. 1 = identique."""
    n_a = emb_a / (np.linalg.norm(emb_a) + 1e-10)
    n_b = emb_b / (np.linalg.norm(emb_b) + 1e-10)
    dist = float(np.clip((1 - np.dot(n_a, n_b)) / 2, 0.0, 1.0))
    return round(1.0 - dist, 4)


def _compute_weights(
    severity:        str,
    geo_ok:          bool,
    texture_ok:      bool,
    ear_ok:          bool,
    is_partial:      bool,
) -> dict:
    """
    Calcule les poids finaux en tenant compte :
      - des composantes disponibles (None → exclu)
      - du bonus oreilles/géo si visage partiel
      - du malus texture si sévérité low
    Retourne un dict {arcface, geometry, texture, ear} normalisé (somme = 1).
    """
    base = _BASE_WEIGHTS.get(severity, _BASE_WEIGHTS["medium"])
    w = {
        "arcface":  base[0],
        "geometry": base[1] if geo_ok     else 0.0,
        "texture":  base[2] if texture_ok else 0.0,
        "ear":      base[3] if ear_ok     else 0.0,
    }

    # Bonus si visage partiel
    if is_partial and ear_ok:
        bonus = min(EAR_PARTIAL_BONUS, w["arcface"] * 0.3)
        w["ear"]     += bonus
        w["arcface"] -= bonus
    if is_partial and geo_ok:
        bonus_geo = min(GEO_PARTIAL_BONUS, w["arcface"] * 0.2)
        w["geometry"] += bonus_geo
        w["arcface"]  -= bonus_geo

    # Malus texture si low severity
    if severity == "low" and texture_ok:
        malus = min(TEX_LOW_MALUS, w["texture"] * 0.4)
        w["texture"] -= malus
        # Redistribue le malus vers géométrie ou arcface
        if geo_ok:
            w["geometry"] += malus
        else:
            w["arcface"]  += malus

    # Clip zéro
    w = {k: max(0.0, v) for k, v in w.items()}

    # Normalisation L1
    total = sum(w.values())
    if total < 1e-8:
        return {"arcface": 1.0, "geometry": 0.0, "texture": 0.0, "ear": 0.0}
    return {k: round(v / total, 4) for k, v in w.items()}


# ══════════════════════════════════════════════════════════════════════════════
#  FUSION PRINCIPALE — 4 composantes
# ══════════════════════════════════════════════════════════════════════════════

def fuse(
    arcface_sim:      float,
    geo_score:        Optional[float],
    texture_score:    Optional[float],
    quality_severity: str,
    ear_score:        Optional[float] = None,
    is_partial: bool = False,
    yaw_max: float = 0.0,
) -> dict:
    """
    Fusionne jusqu'à 4 scores en un verdict final.

    Args:
        arcface_sim      : similarité ArcFace [0-1]
        geo_score        : score géométrie MediaPipe [0-1] ou None
        texture_score    : score texture Gabor [0-1] ou None
        quality_severity : "high" / "medium" / "low"
        ear_score        : score oreilles HOG+Gabor [0-1] ou None
        is_partial       : True si au moins un visage est 3/4 ou partiel

    Retourne :
        {
            "fused_score"  : float [0-1]
            "match"        : bool
            "confidence"   : float (0-100%)
            "threshold"    : seuil de match utilisé
            "method"       : chaîne descriptive (ex: "arcface+geometry+texture+ear")
            "detail"       : décomposition complète
        }
    """
    threshold = THRESHOLDS_SIM.get(quality_severity, THRESHOLDS_SIM["medium"])

    # [FIX-5] |yaw| > 90° = profil/dos → géo frontale inutilisable, on l'exclut
    if abs(yaw_max) > 90.0 and geo_score is not None:
        geo_score = None

    # Calcul des poids adaptatifs
    weights = _compute_weights(
        severity   = quality_severity,
        geo_ok     = geo_score     is not None,
        texture_ok = texture_score is not None,
        ear_ok     = ear_score     is not None,
        is_partial = is_partial,
    )

    # Score fusionné
    scores = {
        "arcface":  arcface_sim,
        "geometry": geo_score     or 0.0,
        "texture":  texture_score or 0.0,
        "ear":      ear_score     or 0.0,
    }
    fused = sum(weights[k] * scores[k] for k in weights)
    fused = round(float(np.clip(fused, 0.0, 1.0)), 4)

    # Méthode
    parts = ["arcface"]
    if geo_score     is not None: parts.append("geometry")
    if texture_score is not None: parts.append("texture")
    if ear_score     is not None: parts.append("ear")

    # Ajustement seuil si visage partiel
    effective_threshold = threshold
    if is_partial:
        effective_threshold = round(threshold - 0.03, 4)   # -3pts de tolérance

    match = bool(fused >= effective_threshold)

    return {
        "fused_score":  fused,
        "match":        match,
        "confidence":   round(fused * 100, 1),
        "threshold":    effective_threshold,
        "method":       "+".join(parts),
        "detail": {
            "arcface_sim":    arcface_sim,
            "geo_score":      geo_score,
            "texture_score":  texture_score,
            "ear_score":      ear_score,
            "weights_used":   weights,
            "quality":        quality_severity,
            "is_partial":     is_partial,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CALIBRATION RUNTIME — mise à jour des seuils depuis regression_tests
# ══════════════════════════════════════════════════════════════════════════════

def update_thresholds(calibrated: dict) -> None:
    """
    Met à jour les seuils de match à partir des résultats de calibration.
    calibrated : {"high": 0.61, "medium": 0.54, "low": 0.49}
    Appelé automatiquement si regression_tests.py trouve un seuil Youden.
    """
    global THRESHOLDS_SIM
    for sev, val in calibrated.items():
        if sev in THRESHOLDS_SIM and isinstance(val, float):
            THRESHOLDS_SIM[sev] = round(val, 4)


def get_thresholds() -> dict:
    """Retourne les seuils actuels (utile pour le dashboard)."""
    return dict(THRESHOLDS_SIM)

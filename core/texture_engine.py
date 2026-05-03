"""
core/texture_engine.py  —  OMNI-RECO v2.0
Texture Engine — Analyse microscopique de la peau via filtres de Gabor

Objectif : différencier deux visages visuellement proches (sosies)
en analysant la texture de la peau à plusieurs échelles et orientations.

Algorithme :
  1. Isolation de la ROI peau (zones joues, front, nez) depuis les landmarks
  2. Banque de filtres de Gabor (5 fréquences × 8 orientations = 40 filtres)
  3. Extraction features : énergie + cohérence par filtre
  4. Normalisation → vecteur de texture 80D
  5. Comparaison cosinus entre deux signatures texture

Références scientifiques :
  - Gabor J. (1946) : filtres de détection de texture multi-échelle
  - Manjunath & Ma (1996) : texture retrieval avec filtres Gabor
  - LBP (Ojala 2002) : alternative utilisée en fallback

Usage :
  from core.texture_engine import TextureEngine
  te = TextureEngine()
  sig = te.extract(image, landmarks_pts)
  score = te.compare(sig_a, sig_b)  → float [0-1]
"""

import io
import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES — Banque de filtres de Gabor
# ══════════════════════════════════════════════════════════════════════════════

# Fréquences spatiales (cycles/pixel) — du macro au microscopique
GABOR_FREQUENCIES = [0.05, 0.10, 0.20, 0.35, 0.50]

# Orientations (radians) — 8 directions couvrant 0→π
GABOR_ORIENTATIONS = [k * np.pi / 8 for k in range(8)]

# Taille du kernel Gabor
GABOR_KSIZE = 31

# Paramètres Gabor
GABOR_SIGMA     = 4.0    # largeur gaussienne
GABOR_GAMMA     = 0.5    # ellipticité (0.5 = ellipse horizontale)
GABOR_BANDWIDTH = 1.0    # bande passante (octaves)


# ══════════════════════════════════════════════════════════════════════════════
#  INDEX MEDIAPIPE — Zones peau (joues, front, nez)
# ══════════════════════════════════════════════════════════════════════════════

# Joue gauche (zone charnue, riche en texture)
SKIN_CHEEK_LEFT  = [116, 111, 117, 118, 119, 120, 121, 128, 126, 142, 36, 205, 187, 123]
# Joue droite
SKIN_CHEEK_RIGHT = [345, 340, 346, 347, 348, 349, 350, 357, 355, 371, 266, 425, 411, 352]
# Front centre
SKIN_FOREHEAD    = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                    397, 365, 379, 378, 400, 377, 152]
# Nez (arête centrale — texture fine)
SKIN_NOSE_BRIDGE = [6, 197, 195, 5, 4]

# Toutes les zones combinées
ALL_SKIN_ZONES = {
    "cheek_left":  SKIN_CHEEK_LEFT,
    "cheek_right": SKIN_CHEEK_RIGHT,
    "forehead":    SKIN_FOREHEAD,
    "nose_bridge": SKIN_NOSE_BRIDGE,
}


# ══════════════════════════════════════════════════════════════════════════════
#  GABOR BANK
# ══════════════════════════════════════════════════════════════════════════════

def build_gabor_bank() -> list:
    """
    Construit la banque de 40 filtres Gabor (5 freq × 8 orientations).
    Retourne une liste de kernels numpy float32.
    Cached en mémoire à la création de TextureEngine.
    """
    bank = []
    for freq in GABOR_FREQUENCIES:
        for theta in GABOR_ORIENTATIONS:
            kernel = cv2.getGaborKernel(
                ksize=(GABOR_KSIZE, GABOR_KSIZE),
                sigma=GABOR_SIGMA,
                theta=theta,
                lambd=1.0 / freq,      # longueur d'onde = 1 / fréquence
                gamma=GABOR_GAMMA,
                psi=0,
                ktype=cv2.CV_32F
            )
            # Normalisation L1 du kernel
            kernel /= (kernel.sum() + 1e-8)
            bank.append(kernel)
    return bank


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_gray(image: Union[str, Path, bytes, io.BytesIO]) -> Optional[np.ndarray]:
    """Charge et convertit en niveaux de gris."""
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
    else:
        raw = image.read() if hasattr(image, "read") else image
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _load_bgr(image: Union[str, Path, bytes, io.BytesIO]) -> Optional[np.ndarray]:
    if isinstance(image, (str, Path)):
        return cv2.imread(str(image))
    raw = image.read() if hasattr(image, "read") else image
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _extract_roi_from_landmarks(
    gray: np.ndarray,
    pts: np.ndarray,
    zone_indices: list,
    padding: int = 4,
) -> Optional[np.ndarray]:
    """
    Extrait une ROI rectangulaire à partir d'un ensemble de landmarks.
    pts : array (N, 3) de coordonnées pixel [x, y, z].
    Retourne un patch numpy (gray) ou None si trop petit.
    """
    if pts is None or len(zone_indices) < 3:
        return None
    zone_pts = pts[zone_indices, :2].astype(int)
    x1, y1   = zone_pts[:, 0].min() - padding, zone_pts[:, 1].min() - padding
    x2, y2   = zone_pts[:, 0].max() + padding, zone_pts[:, 1].max() + padding
    h_img, w_img = gray.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    roi = gray[y1:y2, x1:x2]
    if roi.shape[0] < 16 or roi.shape[1] < 16:
        return None
    return roi


def _gabor_features_from_patch(patch: np.ndarray, bank: list) -> np.ndarray:
    """
    Applique la banque Gabor sur un patch et retourne un vecteur de features.
    Pour chaque filtre : [énergie, cohérence] → vecteur 2 × 40 = 80D.

    Énergie    = mean(|réponse|)       → intensité de la texture à cette fréq/orientation
    Cohérence  = std(|réponse|) / mean → régularité de la texture
    """
    patch_f = patch.astype(np.float32) / 255.0
    features = []
    for kernel in bank:
        resp = cv2.filter2D(patch_f, cv2.CV_32F, kernel)
        magnitude = np.abs(resp)
        energy    = float(magnitude.mean())
        mean_mag  = energy + 1e-8
        coherence = float(magnitude.std() / mean_mag)
        features.extend([energy, coherence])
    return np.array(features, dtype=np.float32)


def _lbp_features(patch: np.ndarray, P: int = 8, R: int = 1) -> np.ndarray:
    """
    LBP (Local Binary Pattern) — fallback si Gabor échoue ou zone trop petite.
    Retourne un histogramme normalisé 59D (LBP uniforme).
    """
    try:
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(patch, P=P, R=R, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=P + 2, range=(0, P + 2), density=True)
        return hist.astype(np.float32)
    except ImportError:
        # Fallback manuel si scikit-image absent
        return np.zeros(P + 2, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  CLASSE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

class TextureEngine:
    """
    Extrait et compare des signatures de texture cutanée.
    Utilise la banque de filtres Gabor + LBP en fallback.

    Taille de la signature : 4 zones × 80 features Gabor = 320D
    (+ 4 × 10 LBP = 40D en fallback)
    """

    def __init__(self):
        self._bank = build_gabor_bank()   # 40 kernels Gabor
        self._n_filters = len(self._bank)  # 40

    # ── Extraction principale ─────────────────────────────────────────────────

    def extract(
        self,
        image: Union[str, Path, bytes, io.BytesIO],
        landmarks_pts: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Extrait la signature texture d'un visage.

        Args:
            image          : image source (chemin, bytes, BytesIO)
            landmarks_pts  : array (478, 3) depuis geometry_engine.py
                             Si None → analyse globale sur image entière

        Retourne :
            {
                "ok": True,
                "signature": np.ndarray (320D ou 80D si global),
                "zone_features": { "cheek_left": array80D, ... },
                "method": "gabor_zonal" | "gabor_global",
                "quality": float [0-1]
            }
        """
        gray = _load_gray(image)
        if gray is None:
            return {"ok": False, "reason": "Image illisible"}

        # ── Mode zonal (landmarks disponibles) ───────────────────────────────
        if landmarks_pts is not None and len(landmarks_pts) >= 468:
            return self._extract_zonal(gray, landmarks_pts)

        # ── Mode global (pas de landmarks) ───────────────────────────────────
        return self._extract_global(gray)

    def _extract_zonal(self, gray: np.ndarray, pts: np.ndarray) -> dict:
        """Extraction zone par zone avec les landmarks MediaPipe."""
        zone_features = {}
        valid_zones   = []

        for zone_name, zone_idx in ALL_SKIN_ZONES.items():
            roi = _extract_roi_from_landmarks(gray, pts, zone_idx)
            if roi is None:
                continue
            # Resize ROI à 64×64 pour homogénéiser
            roi_r = cv2.resize(roi, (128, 128), interpolation=cv2.INTER_CUBIC)  # [FIX-4] 128px homogène
            feats = _gabor_features_from_patch(roi_r, self._bank)  # 80D
            zone_features[zone_name] = feats
            valid_zones.append(zone_name)

        if not zone_features:
            return self._extract_global(gray)

        # Concaténation de toutes les zones → vecteur 320D max
        sig = np.concatenate(list(zone_features.values()))
        # Normalisation L2
        sig = sig / (np.linalg.norm(sig) + 1e-10)

        quality = len(valid_zones) / len(ALL_SKIN_ZONES)

        return {
            "ok":            True,
            "signature":     sig,
            "zone_features": zone_features,
            "zones_used":    valid_zones,
            "method":        "gabor_zonal",
            "dim":           len(sig),
            "quality":       round(quality, 3),
        }

    def _extract_global(self, gray: np.ndarray) -> dict:
        """Extraction sur l'image entière (fallback sans landmarks)."""
        h, w = gray.shape[:2]
        # Crop centre (évite les bords inutiles)
        margin_x = w // 6
        margin_y = h // 6
        roi  = gray[margin_y:h - margin_y, margin_x:w - margin_x]
        roi_r = cv2.resize(roi, (128, 128), interpolation=cv2.INTER_CUBIC)  # [FIX-4] 128px homogène
        feats = _gabor_features_from_patch(roi_r, self._bank)  # 80D
        sig   = feats / (np.linalg.norm(feats) + 1e-10)

        return {
            "ok":            True,
            "signature":     sig,
            "zone_features": {"global": feats},
            "zones_used":    ["global"],
            "method":        "gabor_global",
            "dim":           len(sig),
            "quality":       0.5,   # qualité partielle sans localisation
        }

    # ── Comparaison ──────────────────────────────────────────────────────────

    def compare(self, sig_a: dict, sig_b: dict) -> dict:
        """
        Compare deux signatures texture.

        Stratégie :
          - Si les deux ont des zones en commun → compare zone par zone, moyenne pondérée
          - Sinon → similarité cosinus globale

        Retourne :
            {
                "ok": True,
                "score": float [0-1],    # 1 = identique
                "detail": { zone: score },
                "method": str
            }
        """
        if not sig_a.get("ok") or not sig_b.get("ok"):
            return {"ok": False, "score": 0.0}

        # Poids par zone (joues = plus discriminantes, nez = moins)
        ZONE_WEIGHTS = {
            "cheek_left":  3.0,
            "cheek_right": 3.0,
            "forehead":    2.0,
            "nose_bridge": 1.0,
            "global":      1.5,
        }

        zf_a = sig_a.get("zone_features", {})
        zf_b = sig_b.get("zone_features", {})
        common_zones = set(zf_a.keys()) & set(zf_b.keys())

        if not common_zones:
            # Fallback : cosinus sur signatures globales
            s = _cosine_sim(sig_a["signature"], sig_b["signature"])
            return {"ok": True, "score": round(s, 4),
                    "detail": {}, "method": "cosine_global"}

        total_w = 0.0
        total_s = 0.0
        detail  = {}

        for zone in common_zones:
            w = ZONE_WEIGHTS.get(zone, 1.0)
            s = _cosine_sim(zf_a[zone], zf_b[zone])
            detail[zone] = round(s, 4)
            total_s += w * s
            total_w += w

        global_score = round(total_s / total_w, 4) if total_w > 0 else 0.0

        return {
            "ok":     True,
            "score":  global_score,
            "detail": detail,
            "zones":  list(common_zones),
            "method": "gabor_zonal",
        }


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Similarité cosinus normalisée [0-1]."""
    n_a = a / (np.linalg.norm(a) + 1e-10)
    n_b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.clip((np.dot(n_a, n_b) + 1) / 2, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLETON
# ══════════════════════════════════════════════════════════════════════════════

_TEXTURE_ENGINE: Optional[TextureEngine] = None

def get_texture_engine() -> TextureEngine:
    global _TEXTURE_ENGINE
    if _TEXTURE_ENGINE is None:
        _TEXTURE_ENGINE = TextureEngine()
    return _TEXTURE_ENGINE

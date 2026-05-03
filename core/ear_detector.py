"""
core/ear_detector.py  —  OMNI-RECO v2.0
Ear Biometrics — Détection & extraction de signature auriculaire

Zone oreille : très discriminante (surface ~ fingerprint), quasi-invariante
à l'expression faciale, et ignorée par 99% des systèmes de reco faciale.
Exploitable dès que le visage est de profil ou 3/4 (yaw > 20°).

Pipeline :
  1. Détection via YOLOv8-pose (keypoints 3=oreille_G, 4=oreille_D)
  2. Crop + align de la ROI oreille
  3. Extraction HOG (Histogram of Oriented Gradients) → vecteur 324D
  4. Extraction Gabor auriculaire (5 freq × 6 orient) → 60D
  5. Signature combinée normalisée L2

Fallback si YOLOv8 absent :
  - Estimation position oreille via landmarks MediaPipe (pts 127/356 tempes)
  - Moins précis mais fonctionnel

Référence scientifique :
  - Hurley et al. (2002) : force de la biométrie auriculaire
  - Abaza & Ross (2010) : ear recognition survey
"""

import io
import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional
from dataclasses import dataclass, field

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

# Keypoints COCO dans YOLOv8-pose
KP_LEFT_EAR  = 3   # keypoint index oreille gauche
KP_RIGHT_EAR = 4   # keypoint index oreille droite

# Taille de la ROI normalisée pour extraction
EAR_ROI_SIZE = (64, 128)   # (largeur, hauteur) — oreille est plus haute que large

# Paramètres HOG
HOG_PARAMS = dict(
    winSize    = (64, 128),
    blockSize  = (16, 16),
    blockStride= (8, 8),
    cellSize   = (8, 8),
    nbins      = 9,
)

# Paramètres Gabor auriculaire
EAR_GABOR_FREQ  = [0.08, 0.16, 0.25, 0.35, 0.45]
EAR_GABOR_THETA = [k * np.pi / 6 for k in range(6)]   # 6 orientations

# Padding autour du keypoint pour le crop de la ROI
EAR_CROP_PADDING_RATIO = 1.8   # ×  la distance nez-oreille estimée

# MediaPipe : index des tempes (fallback sans YOLO)
MP_TEMPLE_LEFT  = 127
MP_TEMPLE_RIGHT = 356
MP_EAR_LEFT     = [234, 93, 132, 58, 172, 136, 150, 149, 176]    # contour lobe gauche
MP_EAR_RIGHT    = [454, 323, 361, 288, 397, 365, 379, 378, 400]   # contour lobe droit


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EarROI:
    side:    str              # "left" | "right"
    roi_bgr: np.ndarray       # crop BGR uint8
    bbox:    tuple            # (x1, y1, x2, y2) dans l'image originale
    conf:    float = 1.0      # confiance de détection YOLO [0-1]
    method:  str  = "yolo"    # "yolo" | "mediapipe_fallback"


@dataclass
class EarSignature:
    ok:         bool
    side:       str = ""
    hog_vec:    Optional[np.ndarray] = None     # 324D
    gabor_vec:  Optional[np.ndarray] = None     # 60D
    signature:  Optional[np.ndarray] = None     # 384D concaténé + norm L2
    conf:       float = 0.0
    method:     str   = ""
    reason:     str   = ""


# ══════════════════════════════════════════════════════════════════════════════
#  BANQUE GABOR AURICULAIRE
# ══════════════════════════════════════════════════════════════════════════════

def _build_ear_gabor_bank() -> list:
    bank = []
    for freq in EAR_GABOR_FREQ:
        for theta in EAR_GABOR_THETA:
            k = cv2.getGaborKernel(
                ksize=(21, 21),
                sigma=3.0,
                theta=theta,
                lambd=1.0 / (freq + 1e-8),
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F
            )
            k /= (k.sum() + 1e-8)
            bank.append(k)
    return bank

_EAR_GABOR_BANK = _build_ear_gabor_bank()   # 30 filtres


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS IMAGE
# ══════════════════════════════════════════════════════════════════════════════

def _load_bgr(image: Union[str, Path, bytes, io.BytesIO]) -> Optional[np.ndarray]:
    if isinstance(image, (str, Path)):
        return cv2.imread(str(image))
    raw = image.read() if hasattr(image, "read") else image
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if hasattr(image, "seek"):
        image.seek(0)
    return img


def _crop_ear_roi(img_bgr: np.ndarray, cx: int, cy: int, radius: int) -> Optional[np.ndarray]:
    """Crop une zone carrée centrée sur (cx, cy) avec rayon `radius`."""
    h, w = img_bgr.shape[:2]
    x1   = max(0, cx - radius)
    y1   = max(0, cy - int(radius * 1.6))   # l'oreille est plus haute
    x2   = min(w, cx + radius)
    y2   = min(h, cy + int(radius * 0.8))
    if (x2 - x1) < 16 or (y2 - y1) < 16:
        return None
    roi = img_bgr[y1:y2, x1:x2]
    return cv2.resize(roi, EAR_ROI_SIZE, interpolation=cv2.INTER_CUBIC)


# ══════════════════════════════════════════════════════════════════════════════
#  DÉTECTION YOLO
# ══════════════════════════════════════════════════════════════════════════════

class YOLOPoseDetector:
    """
    Détecte les keypoints corps entier via YOLOv8-pose.
    Extrait spécifiquement les keypoints oreille gauche (3) et droite (4).
    Lazy-load du modèle.
    """

    _MODEL_PATH = Path.home() / ".omni_reco" / "models" / "yolov8n-pose.pt"
    _MODEL_URL  = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt"

    def __init__(self):
        self._model  = None
        self._ready  = False

    def _ensure_model(self, log_fn) -> bool:
        if self._MODEL_PATH.exists():
            return True
        log_fn(f"[EarDetector] Téléchargement YOLOv8-pose ({self._MODEL_URL})")
        self._MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            import urllib.request
            urllib.request.urlretrieve(self._MODEL_URL, str(self._MODEL_PATH))
            log_fn("[EarDetector] YOLOv8-pose téléchargé ✓")
            return True
        except Exception as e:
            log_fn(f"[EarDetector] Téléchargement échoué : {e} → fallback MediaPipe")
            return False

    def load(self, log_fn) -> bool:
        if self._ready:
            return True
        if not self._ensure_model(log_fn):
            return False
        try:
            from ultralytics import YOLO
            # [FIX-8] PyTorch >= 2.6 : weights_only=True par défaut bloque PoseModel
            # On enregistre les globals ultralytics comme sûrs avant le chargement
            try:
                import torch
                import torch.serialization as _ts
                _torch_ver = tuple(int(x) for x in torch.__version__.split(".")[:2])
                if _torch_ver >= (2, 6):
                    try:
                        from ultralytics.nn.tasks import PoseModel, DetectionModel, BaseModel
                        _ts.add_safe_globals([PoseModel, DetectionModel, BaseModel])
                    except Exception:
                        pass
                    # Fallback : forcer weights_only=False via monkey-patch temporaire
                    _orig_load = torch.load
                    def _patched_load(*args, **kwargs):
                        kwargs.setdefault("weights_only", False)
                        return _orig_load(*args, **kwargs)
                    torch.load = _patched_load
                    self._model = YOLO(str(self._MODEL_PATH))
                    torch.load = _orig_load  # restaurer immédiatement
                else:
                    self._model = YOLO(str(self._MODEL_PATH))
            except Exception:
                self._model = YOLO(str(self._MODEL_PATH))
            self._ready = True
            log_fn("[EarDetector] YOLOv8-pose chargé ✓")
            return True
        except ImportError:
            log_fn("[EarDetector] ultralytics non installé → fallback MediaPipe")
            return False
        except Exception as e:
            log_fn(f"[EarDetector] Erreur chargement YOLO : {e}")
            return False

    def detect_ears(
        self,
        img_bgr: np.ndarray,
        log_fn,
        conf_threshold: float = 0.4
    ) -> list:
        """
        Retourne une liste d'EarROI (au plus 2 : gauche + droite).
        """
        if not self._ready:
            return []
        try:
            results  = self._model(img_bgr, verbose=False)[0]
            keypoints = results.keypoints
            if keypoints is None or keypoints.xy is None:
                return []

            ear_rois = []
            for person_kps in keypoints.xy.cpu().numpy():
                # person_kps : (17, 2) en pixels
                conf_kps = keypoints.conf.cpu().numpy()  # (N_persons, 17)

                for side, kp_idx in [("left", KP_LEFT_EAR), ("right", KP_RIGHT_EAR)]:
                    kp = person_kps[kp_idx]
                    # Confiance du keypoint (si disponible)
                    kp_conf = float(conf_kps[0, kp_idx]) if conf_kps is not None else 1.0
                    if kp_conf < conf_threshold:
                        continue
                    cx, cy = int(kp[0]), int(kp[1])
                    if cx == 0 and cy == 0:
                        continue

                    # Rayon de crop : ~4% de la largeur image
                    radius = max(20, int(img_bgr.shape[1] * 0.04))
                    roi    = _crop_ear_roi(img_bgr, cx, cy, radius)
                    if roi is None:
                        continue

                    x1 = max(0, cx - radius)
                    y1 = max(0, cy - int(radius * 1.6))
                    ear_rois.append(EarROI(
                        side=side, roi_bgr=roi,
                        bbox=(x1, y1, x1 + EAR_ROI_SIZE[0], y1 + EAR_ROI_SIZE[1]),
                        conf=kp_conf, method="yolo"
                    ))
            return ear_rois

        except Exception as e:
            log_fn(f"[EarDetector] Erreur inférence YOLO : {e}")
            return []


# ══════════════════════════════════════════════════════════════════════════════
#  FALLBACK : ESTIMATION VIA MEDIAPIPE
# ══════════════════════════════════════════════════════════════════════════════

def _ear_rois_from_mediapipe(
    img_bgr: np.ndarray,
    landmarks_pts: np.ndarray,
    log_fn,
) -> list:
    """
    Estime la zone oreille depuis les landmarks tempes MediaPipe.
    Moins précis que YOLO mais ne nécessite aucune dépendance supplémentaire.
    landmarks_pts : (478, 3) array de coords pixel.
    """
    if landmarks_pts is None or len(landmarks_pts) < 400:
        return []

    ear_rois = []
    h_img, w_img = img_bgr.shape[:2]

    for side, temple_idx, ear_indices in [
        ("left",  MP_TEMPLE_LEFT,  MP_EAR_LEFT),
        ("right", MP_TEMPLE_RIGHT, MP_EAR_RIGHT),
    ]:
        temple = landmarks_pts[temple_idx, :2].astype(int)
        cx, cy = temple[0], temple[1]

        # Estime le rayon depuis l'écart entre les points du contour oreille
        ear_pts = landmarks_pts[ear_indices, :2].astype(int)
        radius  = max(20, int(
            np.max([np.linalg.norm(ear_pts[i] - ear_pts[j])
                    for i in range(len(ear_pts))
                    for j in range(i+1, len(ear_pts))]) / 1.5
        ))

        roi = _crop_ear_roi(img_bgr, cx, cy, radius)
        if roi is None:
            continue

        ear_rois.append(EarROI(
            side=side, roi_bgr=roi,
            bbox=(max(0, cx-radius), max(0, cy-int(radius*1.6)),
                  min(w_img, cx+radius), min(h_img, cy+int(radius*0.8))),
            conf=0.6,
            method="mediapipe_fallback"
        ))
        log_fn(f"[EarDetector] Oreille {side} estimée via MediaPipe (cx={cx}, cy={cy})")

    return ear_rois


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACTION DE SIGNATURE
# ══════════════════════════════════════════════════════════════════════════════

def _hog_features(roi_bgr: np.ndarray) -> np.ndarray:
    """
    HOG sur ROI 64×128 → vecteur 3780D réduit à 324D par PCA lite (moyenne blocs).
    On utilise la version OpenCV HOGDescriptor pour vitesse.
    """
    gray    = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 128))
    hog     = cv2.HOGDescriptor(
        _winSize    = (64, 128),
        _blockSize  = (16, 16),
        _blockStride= (8, 8),
        _cellSize   = (8, 8),
        _nbins      = 9
    )
    desc = hog.compute(resized)
    # desc shape : (3780, 1) → flatten
    desc = desc.flatten()
    # Réduction dimensionnelle simple : reshape en blocs et moyenner
    # 3780 = 420 blocs × 9 bins → on garde les 36 cellules × 9 bins = 324
    n_cells = (64 // 8) * (128 // 8)   # = 128 cellules
    cell_hists = desc[:n_cells * 9].reshape(n_cells, 9)
    # Réduction : moyenne par groupe de 4 cellules → 32 × 9 = 288 + norm stats
    reduced = cell_hists.reshape(-1, 4, 9).mean(axis=1).flatten()   # 32×9 = 288
    # Normalisation L2
    reduced /= (np.linalg.norm(reduced) + 1e-10)
    return reduced.astype(np.float32)


def _gabor_ear_features(roi_bgr: np.ndarray) -> np.ndarray:
    """Gabor bank auriculaire sur ROI → vecteur 60D (30 filtres × 2 stats)."""
    gray  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    patch = cv2.resize(gray, (64, 64)).astype(np.float32) / 255.0
    feats = []
    for kernel in _EAR_GABOR_BANK:
        resp  = cv2.filter2D(patch, cv2.CV_32F, kernel)
        mag   = np.abs(resp)
        feats.extend([mag.mean(), mag.std() / (mag.mean() + 1e-8)])
    vec = np.array(feats, dtype=np.float32)
    vec /= (np.linalg.norm(vec) + 1e-10)
    return vec


def _extract_signature(ear_roi: EarROI) -> EarSignature:
    """Extrait HOG + Gabor depuis une EarROI → EarSignature."""
    try:
        hog_vec   = _hog_features(ear_roi.roi_bgr)
        gabor_vec = _gabor_ear_features(ear_roi.roi_bgr)
        sig       = np.concatenate([hog_vec, gabor_vec])
        sig       = sig / (np.linalg.norm(sig) + 1e-10)
        return EarSignature(
            ok=True, side=ear_roi.side,
            hog_vec=hog_vec, gabor_vec=gabor_vec,
            signature=sig, conf=ear_roi.conf,
            method=ear_roi.method
        )
    except Exception as e:
        return EarSignature(ok=False, reason=str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  COMPARAISON
# ══════════════════════════════════════════════════════════════════════════════

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    n_a = a / (np.linalg.norm(a) + 1e-10)
    n_b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.clip((np.dot(n_a, n_b) + 1) / 2, 0.0, 1.0))


def compare_ear_signatures(sigs_a: list, sigs_b: list) -> dict:
    """
    Compare des listes de signatures auriculaires.
    Stratégie : compare les oreilles du même côté si disponibles.
    Retourne {"ok": bool, "score": float, "detail": dict}.
    """
    if not sigs_a or not sigs_b:
        return {"ok": False, "score": None, "detail": {}}

    # Indexer par côté
    idx_a = {s.side: s for s in sigs_a if s.ok}
    idx_b = {s.side: s for s in sigs_b if s.ok}
    common = set(idx_a.keys()) & set(idx_b.keys())

    if not common:
        # Fallback : compare best sig_a vs best sig_b
        best_a = max(sigs_a, key=lambda s: s.conf if s.ok else 0)
        best_b = max(sigs_b, key=lambda s: s.conf if s.ok else 0)
        if not best_a.ok or not best_b.ok:
            return {"ok": False, "score": None, "detail": {}}
        score = _cosine_sim(best_a.signature, best_b.signature)
        return {"ok": True, "score": round(score, 4),
                "detail": {"cross_side": True}, "method": "ear_crossside"}

    scores = {}
    for side in common:
        scores[side] = round(_cosine_sim(
            idx_a[side].signature, idx_b[side].signature
        ), 4)

    # Moyenne pondérée (oreilles plus visibles = plus de confiance)
    total_w = sum(idx_a[s].conf * idx_b[s].conf for s in common)
    total_s = sum(scores[s] * idx_a[s].conf * idx_b[s].conf for s in common)
    final   = round(total_s / (total_w + 1e-10), 4)

    return {
        "ok":     True,
        "score":  final,
        "detail": scores,
        "method": "ear_zonal",
        "sides":  list(common),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CLASSE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

class EarDetector:
    """
    Interface principale du module auriculaire.
    Combine YOLO-pose (primaire) + MediaPipe (fallback).

    Usage :
        ed = EarDetector()
        sigs = ed.extract(image, landmarks_pts=geo_sig["landmarks_px"])
        result = ed.compare(sigs_a, sigs_b)
    """

    def __init__(self):
        self._yolo   = YOLOPoseDetector()
        self._log_fn = print

    def set_log_fn(self, fn):
        self._log_fn = fn

    def extract(
        self,
        image:          Union[str, Path, bytes, io.BytesIO],
        landmarks_pts:  Optional[np.ndarray] = None,
        yaw:            float = 0.0,
    ) -> list:
        """
        Détecte les oreilles et extrait leurs signatures.

        Args:
            image         : image source
            landmarks_pts : (478, 3) depuis geometry_engine (optionnel)
            yaw           : angle yaw estimé (déclenche l'analyse si > 15°)

        Retourne : liste de EarSignature (0, 1 ou 2 éléments)
        """
        img_bgr = _load_bgr(image)
        if img_bgr is None:
            self._log_fn("[EarDetector] Image illisible")
            return []

        h, w = img_bgr.shape[:2]

        # On analyse les oreilles si yaw > 15° OU si on est en mode full-scan
        if abs(yaw) < 15.0 and landmarks_pts is None:
            self._log_fn(f"[EarDetector] yaw={yaw:.1f}° < 15° — oreilles non analysées")
            return []

        # ── Tentative YOLO ───────────────────────────────────────────────────
        ear_rois = []
        if self._yolo.load(self._log_fn):
            ear_rois = self._yolo.detect_ears(img_bgr, self._log_fn)
            self._log_fn(f"[EarDetector] YOLO : {len(ear_rois)} oreille(s) détectée(s)")

        # ── Fallback MediaPipe ───────────────────────────────────────────────
        if not ear_rois and landmarks_pts is not None:
            ear_rois = _ear_rois_from_mediapipe(img_bgr, landmarks_pts, self._log_fn)

        if not ear_rois:
            self._log_fn("[EarDetector] Aucune oreille détectable")
            return []

        # ── Extraction signatures ────────────────────────────────────────────
        signatures = [_extract_signature(roi) for roi in ear_rois]
        valid = [s for s in signatures if s.ok]
        self._log_fn(f"[EarDetector] {len(valid)} signature(s) auriculaire(s) extraite(s)")
        return valid

    def compare(self, sigs_a: list, sigs_b: list) -> dict:
        return compare_ear_signatures(sigs_a, sigs_b)


# ── Singleton ─────────────────────────────────────────────────────────────────
_EAR_DETECTOR: Optional[EarDetector] = None

def get_ear_detector() -> EarDetector:
    global _EAR_DETECTOR
    if _EAR_DETECTOR is None:
        _EAR_DETECTOR = EarDetector()
    return _EAR_DETECTOR

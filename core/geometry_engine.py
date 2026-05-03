"""
core/geometry_engine.py  —  OMNI-RECO v2.0
Signature Extractor 3D — MediaPipe Face Mesh 468 pts + Iris 468-477

CORRECTIONS vs code de base :
  1. dist_eyes calculé en 3D RÉEL (pas juste norme euclidienne brute) → compensation yaw
  2. Sourcils : index complets (pas juste 105/70), courbure polyligne vs simple distance
  3. Score de fiabilité par zone → sortie "available_regions" pour le fusion_engine
  4. Normalisation sur IPD 3D corrigée par profondeur Z pour les portraits 3/4
  5. Estimation pose (yaw/pitch/roll) via points canoniques pour pondérer les ratios
"""

import io
import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional

# ══════════════════════════════════════════════════════════════════════════════
#  INDEX MEDIAPIPE FACE MESH — Constantes nommées (ne jamais hardcoder inline)
# ══════════════════════════════════════════════════════════════════════════════

# ── Iris (refine_landmarks=True requis) ──────────────────────────────────────
IRIS_LEFT_CENTER  = 468
IRIS_LEFT_RING    = [468, 469, 470, 471, 472]
IRIS_RIGHT_CENTER = 473
IRIS_RIGHT_RING   = [473, 474, 475, 476, 477]

# ── Sourcils ─────────────────────────────────────────────────────────────────
BROW_LEFT  = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]   # gauche (10 pts)
BROW_RIGHT = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]  # droit (10 pts)

# ── Yeux (contour complet) ───────────────────────────────────────────────────
EYE_LEFT  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
EYE_RIGHT = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# ── Points cardinaux yeux (pour EAR - Eye Aspect Ratio) ─────────────────────
EYE_LEFT_INNER   = 133   # coin interne
EYE_LEFT_OUTER   = 33    # coin externe
EYE_LEFT_TOP     = 159   # paupière haute
EYE_LEFT_BOT     = 145   # paupière basse
EYE_RIGHT_INNER  = 362
EYE_RIGHT_OUTER  = 263
EYE_RIGHT_TOP    = 386
EYE_RIGHT_BOT    = 374

# ── Nez ──────────────────────────────────────────────────────────────────────
NOSE_TIP         = 4
NOSE_BRIDGE_TOP  = 6
NOSE_BRIDGE_MID  = 197
NOSE_WING_LEFT   = 129
NOSE_WING_RIGHT  = 358
NOSE_BASE_CENTER = 2

# ── Bouche ───────────────────────────────────────────────────────────────────
MOUTH_LEFT   = 61
MOUTH_RIGHT  = 291
MOUTH_TOP    = 13     # lèvre haute centre
MOUTH_BOT    = 14     # lèvre basse centre

# ── Repères de pose ──────────────────────────────────────────────────────────
CHIN_TIP     = 152
FOREHEAD_TOP = 10
TEMPLE_LEFT  = 127
TEMPLE_RIGHT = 356

# ── Points de référence pose 3D (modèle canonique PnP) ──────────────────────
# Ces points ont des coordonnées 3D "réelles" connues pour solvePnP
PNP_LANDMARKS_IDX = [NOSE_TIP, CHIN_TIP, EYE_LEFT_OUTER, EYE_RIGHT_OUTER,
                     MOUTH_LEFT, MOUTH_RIGHT]
PNP_MODEL_POINTS = np.array([
    [0.0,    0.0,     0.0],    # Pointe du nez
    [0.0,   -63.6,  -12.5],   # Menton
    [-43.3,  32.7,  -26.0],   # Coin oeil gauche
    [ 43.3,  32.7,  -26.0],   # Coin oeil droit
    [-28.9, -28.9,  -24.1],   # Coin bouche gauche
    [ 28.9, -28.9,  -24.1],   # Coin bouche droit
], dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS GÉOMÉTRIQUES
# ══════════════════════════════════════════════════════════════════════════════

def _poly_curvature(pts: np.ndarray) -> float:
    """
    Courbure d'une polyligne de points 3D.
    Ratio : flèche max / corde totale. 0 = ligne droite, 1 = très courbé.
    """
    if len(pts) < 3:
        return 0.0
    chord = np.linalg.norm(pts[-1] - pts[0])
    if chord < 1e-6:
        return 0.0
    # Distance max d'un point à la droite chord
    dists = []
    d_vec = (pts[-1] - pts[0]) / chord
    for p in pts[1:-1]:
        v = p - pts[0]
        proj = np.dot(v, d_vec)
        perp = np.linalg.norm(v - proj * d_vec)
        dists.append(perp)
    return float(max(dists) / chord) if dists else 0.0


def _poly_slope(pts: np.ndarray) -> float:
    """
    Pente de la polyligne (angle en radians entre extrémité gauche et droite).
    Positif = sourcil monte vers l'extérieur, négatif = tombe.
    """
    if len(pts) < 2:
        return 0.0
    dx = pts[-1][0] - pts[0][0]
    dy = pts[-1][1] - pts[0][1]
    return float(np.arctan2(dy, dx))


def _centroid(pts: np.ndarray) -> np.ndarray:
    return pts.mean(axis=0)


def _dist3d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _iris_radius(ring_pts: np.ndarray, center: np.ndarray) -> float:
    """Rayon moyen de l'iris à partir des 4 points du ring + centre."""
    dists = [_dist3d(p, center) for p in ring_pts]
    return float(np.mean(dists)) if dists else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  CLASSE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

class GeometryEngine:
    """
    Extrait une signature géométrique 3D normalisée d'un visage via MediaPipe.
    Thread-safe : instancier un objet par thread.
    """

    def __init__(self, min_detection_confidence: float = 0.5):
        self._ready     = False
        self._face_mesh = None
        try:
            import mediapipe as mp
            face_mesh_module = None

            # Tentative 1 : chemin classique mp.solutions
            try:
                if hasattr(mp, "solutions"):
                    import mediapipe.python.solutions.face_mesh as _fm1
                    face_mesh_module = _fm1
            except Exception:
                pass

            # Tentative 2 : import direct du sous-module
            if face_mesh_module is None:
                try:
                    from mediapipe.python.solutions import face_mesh as _fm2
                    face_mesh_module = _fm2
                except Exception:
                    pass

            # Tentative 3 : importlib
            if face_mesh_module is None:
                try:
                    import importlib
                    face_mesh_module = importlib.import_module(
                        "mediapipe.python.solutions.face_mesh"
                    )
                except Exception:
                    pass

            if face_mesh_module is None:
                print("[GEOMETRY] Impossible de charger FaceMesh.")
                print("[GEOMETRY] Reinstalle mediapipe dans le venv :")
                print("[GEOMETRY]   venv_omni\\Scripts\\activate")
                print("[GEOMETRY]   pip install mediapipe==0.10.9")
                return

            self._face_mesh = face_mesh_module.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
            )
            self._ready = True
            print("[GEOMETRY] FaceMesh initialise avec succes.")

        except ImportError:
            print("[GEOMETRY] mediapipe absent du venv actif.")
            print("[GEOMETRY] Lance pip_install.bat ou :")
            print("[GEOMETRY]   pip install mediapipe==0.10.9")
        except Exception as e:
            print(f"[GEOMETRY] Erreur inattendue : {e}")

    # ── Statut ───────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ── Chargement image ──────────────────────────────────────────────────────

    def _load_bgr(self, image: Union[str, Path, bytes, io.BytesIO]) -> Optional[np.ndarray]:
        if isinstance(image, (str, Path)):
            return cv2.imread(str(image))
        raw = image.read() if hasattr(image, "read") else image
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # ── Extraction landmarks ──────────────────────────────────────────────────

    def _get_lm_array(self, image_bgr: np.ndarray):
        """
        Retourne un array (478, 3) de coordonnées pixel [x, y, z*w]
        ou None si aucun visage.
        CORRECTION v2 : Z est multiplié par img_w (même échelle que X)
        pour que les distances 3D soient cohérentes.
        """
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None, w, h
        lm = results.multi_face_landmarks[0].landmark
        pts = np.array([[p.x * w, p.y * h, p.z * w] for p in lm], dtype=np.float32)
        return pts, w, h

    # ── Estimation de pose (yaw / pitch / roll) ───────────────────────────────

    def _estimate_pose(self, pts: np.ndarray, img_w: int, img_h: int) -> dict:
        """
        solvePnP sur 6 points canoniques → angles euler en degrés.
        Retourne {"yaw", "pitch", "roll", "reliable"}.
        """
        try:
            focal = img_w  # approximation : f ≈ largeur image
            cx, cy = img_w / 2.0, img_h / 2.0
            cam_matrix = np.array([
                [focal, 0, cx],
                [0, focal, cy],
                [0,     0,  1],
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1))
            image_pts = np.array([pts[i, :2] for i in PNP_LANDMARKS_IDX], dtype=np.float64)
            ok, rvec, tvec = cv2.solvePnP(
                PNP_MODEL_POINTS, image_pts, cam_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not ok:
                return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "reliable": False}
            R, _ = cv2.Rodrigues(rvec)
            # Angles d'Euler (ZYX convention)
            sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
            if sy > 1e-6:
                pitch = float(np.degrees(np.arctan2(-R[2, 0], sy)))
                yaw   = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
                roll  = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
            else:
                pitch = float(np.degrees(np.arctan2(-R[2, 0], sy)))
                yaw   = 0.0
                roll  = float(np.degrees(np.arctan2(-R[0, 1], R[1, 1])))
            # [FIX-7] Sanity-check yaw aberrant sur portrait serré
            # Si |yaw| > 90° mais les yeux sont symétriques (visage frontal)
            # → PnP s'est planté à cause du focal mal estimé → forcer yaw=0
            _lm_l = image_pts[2]; _lm_r = image_pts[3]
            _cx   = img_w / 2.0
            _dl   = abs(_lm_l[0] - _cx); _dr = abs(_lm_r[0] - _cx)
            _sym  = min(_dl, _dr) / (max(_dl, _dr) + 1e-6)
            if abs(yaw) > 90.0 and _sym > 0.60:
                yaw = 0.0
            return {"yaw": round(yaw, 2), "pitch": round(pitch, 2),
                    "roll": round(roll, 2), "reliable": True}
        except Exception:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "reliable": False}

    # ── Distance inter-pupillaire corrigée 3D ─────────────────────────────────

    def _ipd_3d(self, pts: np.ndarray, yaw_deg: float) -> float:
        """
        CORRECTION CRITIQUE vs v1 :
        Si yaw > 15°, la distance 2D entre pupilles est compressée.
        On corrige par : IPD_real = IPD_2d / cos(yaw)
        (valable jusqu'à ~45°, au delà on clampe)
        """
        ipd_2d = _dist3d(pts[IRIS_LEFT_CENTER], pts[IRIS_RIGHT_CENTER])
        yaw_rad = np.radians(min(abs(yaw_deg), 45.0))
        correction = np.cos(yaw_rad)
        return float(ipd_2d / correction) if correction > 0.1 else float(ipd_2d)

    # ── Calcul des ratios par zone ────────────────────────────────────────────

    def _ratios_brows(self, pts: np.ndarray, ipd: float) -> dict:
        bl = pts[BROW_LEFT]   # (10, 3)
        br = pts[BROW_RIGHT]
        el_center = pts[IRIS_LEFT_CENTER]
        er_center = pts[IRIS_RIGHT_CENTER]

        curv_l = _poly_curvature(bl)
        curv_r = _poly_curvature(br)
        slope_l = _poly_slope(bl)
        slope_r = _poly_slope(br)
        be_dist_l = _dist3d(_centroid(bl), el_center) / ipd
        be_dist_r = _dist3d(_centroid(br), er_center) / ipd
        asym_curv = abs(curv_l - curv_r) / (max(curv_l, curv_r) + 1e-6)

        return {
            "brow_curvature_left":  round(curv_l, 4),
            "brow_curvature_right": round(curv_r, 4),
            "brow_curvature_asym":  round(asym_curv, 4),
            "brow_slope_left":      round(slope_l, 4),
            "brow_slope_right":     round(slope_r, 4),
            "brow_eye_dist_left":   round(be_dist_l, 4),
            "brow_eye_dist_right":  round(be_dist_r, 4),
        }

    def _ratios_eyes(self, pts: np.ndarray, ipd: float) -> dict:
        """Eye Aspect Ratio (EAR) — invariant à l'identité, stable."""
        # EAR = (vertical_top_bot) / (2 * horizontal_inner_outer)
        v_l = _dist3d(pts[EYE_LEFT_TOP], pts[EYE_LEFT_BOT])
        h_l = _dist3d(pts[EYE_LEFT_INNER], pts[EYE_LEFT_OUTER])
        ear_l = v_l / (2 * h_l + 1e-6)

        v_r = _dist3d(pts[EYE_RIGHT_TOP], pts[EYE_RIGHT_BOT])
        h_r = _dist3d(pts[EYE_RIGHT_INNER], pts[EYE_RIGHT_OUTER])
        ear_r = v_r / (2 * h_r + 1e-6)

        # Iris radius
        ir_l = _iris_radius(pts[IRIS_LEFT_RING[1:]], pts[IRIS_LEFT_CENTER]) / ipd
        ir_r = _iris_radius(pts[IRIS_RIGHT_RING[1:]], pts[IRIS_RIGHT_CENTER]) / ipd

        # Largeur oeil normalisée
        ew_l = h_l / ipd
        ew_r = h_r / ipd

        return {
            "ear_left":          round(ear_l, 4),
            "ear_right":         round(ear_r, 4),
            "ear_asym":          round(abs(ear_l - ear_r) / (max(ear_l, ear_r) + 1e-6), 4),
            "iris_radius_left":  round(ir_l, 4),
            "iris_radius_right": round(ir_r, 4),
            "eye_width_left":    round(ew_l, 4),
            "eye_width_right":   round(ew_r, 4),
        }

    def _ratios_nose(self, pts: np.ndarray, ipd: float) -> dict:
        nose_w = _dist3d(pts[NOSE_WING_LEFT], pts[NOSE_WING_RIGHT]) / ipd
        nose_h = _dist3d(pts[NOSE_BRIDGE_TOP], pts[NOSE_TIP]) / ipd
        nose_base = _dist3d(pts[NOSE_BASE_CENTER], pts[NOSE_TIP]) / ipd
        bridge_w = _dist3d(pts[NOSE_BRIDGE_MID], pts[NOSE_TIP])  # longueur pont
        bridge_ratio = bridge_w / (ipd + 1e-6)

        return {
            "nose_width_ratio":  round(nose_w, 4),
            "nose_height_ratio": round(nose_h, 4),
            "nose_base_ratio":   round(nose_base, 4),
            "nose_bridge_ratio": round(bridge_ratio, 4),
        }

    def _ratios_mouth(self, pts: np.ndarray, ipd: float) -> dict:
        mouth_w = _dist3d(pts[MOUTH_LEFT], pts[MOUTH_RIGHT]) / ipd
        mouth_h = _dist3d(pts[MOUTH_TOP], pts[MOUTH_BOT]) / ipd
        # Triangle nez-bouche-menton
        nose_mouth = _dist3d(pts[NOSE_TIP], pts[MOUTH_TOP]) / ipd
        mouth_chin  = _dist3d(pts[MOUTH_BOT], pts[CHIN_TIP]) / ipd
        midface_tri = nose_mouth / (mouth_chin + 1e-6)

        return {
            "mouth_width_ratio":  round(mouth_w, 4),
            "mouth_height_ratio": round(mouth_h, 4),
            "nose_mouth_dist":    round(nose_mouth, 4),
            "mouth_chin_dist":    round(mouth_chin, 4),
            "midface_triangle":   round(midface_tri, 4),
        }

    # ── Score de fiabilité par zone ───────────────────────────────────────────

    def _zone_quality(self, pts: np.ndarray, pose: dict) -> dict:
        """
        Estime si chaque zone est exploitable selon la pose.
        yaw > 30° → zones latérales dégradées.
        pitch > 20° → zones hautes/basses dégradées.
        Retourne un score 0-1 par zone + liste des zones fiables.
        """
        yaw   = abs(pose.get("yaw", 0.0))
        pitch = abs(pose.get("pitch", 0.0))

        def _score_zone(yaw_limit, pitch_limit):
            s_yaw   = max(0.0, 1.0 - max(0.0, yaw - yaw_limit) / 30.0)
            s_pitch = max(0.0, 1.0 - max(0.0, pitch - pitch_limit) / 20.0)
            return round(s_yaw * s_pitch, 3)

        scores = {
            "brows":  _score_zone(25, 20),
            "eyes":   _score_zone(30, 25),
            "iris":   _score_zone(20, 20),   # iris très sensible au yaw
            "nose":   _score_zone(35, 30),
            "mouth":  _score_zone(30, 25),
        }
        available = [z for z, s in scores.items() if s >= 0.5]
        global_q  = round(float(np.mean(list(scores.values()))), 3)

        return {
            "zone_scores":         scores,
            "available_regions":   available,
            "global_quality":      global_q,
        }

    # ── API publique ──────────────────────────────────────────────────────────

    def extract(self, image: Union[str, Path, bytes, io.BytesIO]) -> dict:
        """
        Point d'entrée unique.
        Retourne un dict complet ou {"ok": False, "reason": str} si échec.
        """
        if not self._ready:
            return {"ok": False, "reason": "MediaPipe non disponible"}

        img_bgr = self._load_bgr(image)
        if img_bgr is None:
            return {"ok": False, "reason": "Image illisible"}

        pts, img_w, img_h = self._get_lm_array(img_bgr)
        if pts is None:
            return {"ok": False, "reason": "Aucun visage détecté"}

        pose = self._estimate_pose(pts, img_w, img_h)
        ipd  = self._ipd_3d(pts, pose["yaw"])

        if ipd < 5.0:
            return {"ok": False, "reason": f"IPD trop faible ({ipd:.1f}px) — visage trop petit ou trop loin"}

        quality = self._zone_quality(pts, pose)

        ratios = {}
        ratios.update(self._ratios_brows(pts, ipd))
        ratios.update(self._ratios_eyes(pts, ipd))
        ratios.update(self._ratios_nose(pts, ipd))
        ratios.update(self._ratios_mouth(pts, ipd))

        return {
            "ok":               True,
            "pose":             pose,
            "scale_ref_ipd":    round(ipd, 2),
            "image_size":       [img_w, img_h],
            "ratios":           ratios,
            "quality":          quality,
            "n_landmarks":      len(pts),
        }

    def compare(self, sig_a: dict, sig_b: dict, weights: Optional[dict] = None) -> dict:
        """
        Compare deux signatures géométriques.
        Retourne un score de similarité global [0-1] et le détail par zone.
        weights : pondération optionnelle par zone. Par défaut : égal.
        """
        if not sig_a.get("ok") or not sig_b.get("ok"):
            return {"ok": False, "score": 0.0, "reason": "Signature invalide"}

        DEFAULT_WEIGHTS = {
            "brow_curvature_left":  1.5,
            "brow_curvature_right": 1.5,
            "brow_eye_dist_left":   2.0,
            "brow_eye_dist_right":  2.0,
            "ear_left":             2.5,
            "ear_right":            2.5,
            "iris_radius_left":     1.5,
            "iris_radius_right":    1.5,
            "nose_width_ratio":     3.0,
            "nose_height_ratio":    2.5,
            "mouth_width_ratio":    2.0,
            "midface_triangle":     3.0,
        }
        w = weights or DEFAULT_WEIGHTS
        ra = sig_a["ratios"]
        rb = sig_b["ratios"]

        # Zones disponibles dans les DEUX signatures
        avail_a = set(sig_a["quality"]["available_regions"])
        avail_b = set(sig_b["quality"]["available_regions"])
        avail_both = avail_a & avail_b

        ZONE_MAP = {
            "brows":  ["brow_curvature_left","brow_curvature_right","brow_eye_dist_left","brow_eye_dist_right"],
            "eyes":   ["ear_left","ear_right","eye_width_left","eye_width_right"],
            "iris":   ["iris_radius_left","iris_radius_right"],
            "nose":   ["nose_width_ratio","nose_height_ratio","nose_bridge_ratio"],
            "mouth":  ["mouth_width_ratio","mouth_height_ratio","midface_triangle"],
        }

        total_w  = 0.0
        total_sc = 0.0
        detail   = {}

        for zone, keys in ZONE_MAP.items():
            if zone not in avail_both:
                detail[zone] = None
                continue
            z_scores = []
            z_w      = []
            for k in keys:
                if k not in ra or k not in rb:
                    continue
                wi = w.get(k, 1.0)
                # Similarité = 1 - |diff| / max(val_a, val_b, 0.001)
                denom = max(abs(ra[k]), abs(rb[k]), 0.001)
                sim   = max(0.0, 1.0 - abs(ra[k] - rb[k]) / denom)
                z_scores.append(sim * wi)
                z_w.append(wi)
            if z_w:
                zone_score = sum(z_scores) / sum(z_w)
                detail[zone] = round(zone_score, 4)
                total_sc += sum(z_scores)
                total_w  += sum(z_w)

        global_score = round(total_sc / total_w, 4) if total_w > 0 else 0.0

        return {
            "ok":           True,
            "score":        global_score,
            "detail":       detail,
            "zones_used":   list(avail_both),
            "zones_missing": list(set(ZONE_MAP.keys()) - avail_both),
        }

    def close(self):
        if self._ready:
            self._face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLETON (même pattern que face_engine.py)
# ══════════════════════════════════════════════════════════════════════════════

_GEO_ENGINE: Optional[GeometryEngine] = None

def get_geometry_engine() -> GeometryEngine:
    global _GEO_ENGINE
    if _GEO_ENGINE is None:
        _GEO_ENGINE = GeometryEngine()
    return _GEO_ENGINE

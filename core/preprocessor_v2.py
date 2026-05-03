"""
core/preprocessor_v2.py  —  OMNI-RECO v2.0
Pipeline de prétraitement image — Version chirurgicale

Étapes :
  1. Lecture & normalisation
  2. ESRGAN super-résolution (si image < seuil de résolution)
  3. CLAHE  (contraste local LAB)
  4. Denoising NlMeans
  5. Sharpening Unsharp Mask
  6. Export BytesIO JPEG 95

Modèle ESRGAN utilisé : realesr-general-x4v3.pth (Real-ESRGAN, x4 upscale)
  → ONNX CPU-only : realesrgan-x4.onnx  (~17 Mo)
  → Auto-téléchargement si absent
  → Fallback : upscale bicubique OpenCV si ONNX non dispo

Design : zéro écriture disque en mode stealth (-S)
"""

import io
import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional, Callable

# ── Seuil en dessous duquel on déclenche ESRGAN ──────────────────────────────
ESRGAN_MIN_PX   = 128   # si min(h,w) < 128px → ESRGAN
BICUBIC_MIN_PX  = 200   # si < 200px sans ESRGAN → upscale bicubique (legacy)
ESRGAN_SCALE    = 4     # facteur × du modèle

# URL du modèle ONNX léger Real-ESRGAN x4 (general, CPU-friendly)
ESRGAN_ONNX_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
    "realesrgan-ncnn-vulkan-20220424-windows.zip"  # fallback zip officiel
)
# Modèle ONNX converti direct (plus simple, ~17 Mo)
ESRGAN_ONNX_URL_DIRECT = (
    "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/"
    "realesrgan-x4.onnx"
)
ESRGAN_ONNX_PATH = Path.home() / ".omni_reco" / "models" / "realesrgan-x4.onnx"


# ══════════════════════════════════════════════════════════════════════════════
#  ESRGAN ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class ESRGANUpscaler:
    """
    Wrapper ONNX pour Real-ESRGAN x4.
    Thread-safe : une instance par process.
    Lazy-loading : le modèle n'est chargé qu'au premier appel upscale().
    """

    def __init__(self, model_path: Path = ESRGAN_ONNX_PATH):
        self.model_path = model_path
        self._session  = None
        self._ready    = False

    def _ensure_model(self, log_fn: Callable):
        if self.model_path.exists():
            return True
        log_fn(f"[ESRGAN] Modèle ONNX introuvable → téléchargement ({ESRGAN_ONNX_PATH})")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import urllib.request
            urllib.request.urlretrieve(ESRGAN_ONNX_URL_DIRECT, str(self.model_path))
            log_fn("[ESRGAN] Modèle téléchargé avec succès.", )
            return True
        except Exception as e:
            log_fn(f"[ESRGAN] Échec téléchargement : {e} — fallback bicubique activé")
            return False

    def _load_session(self, log_fn: Callable) -> bool:
        if self._ready:
            return True
        if not self._ensure_model(log_fn):
            return False
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"]
            )
            self._input_name  = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            self._ready = True
            log_fn("[ESRGAN] Session ONNX chargée — Super-Résolution x4 active.")
            return True
        except Exception as e:
            log_fn(f"[ESRGAN] Erreur chargement ONNX : {e}")
            return False

    def upscale(self, img_bgr: np.ndarray, log_fn: Callable) -> np.ndarray:
        """
        Upscale ×4 via Real-ESRGAN.
        img_bgr : numpy array BGR uint8.
        Retourne l'image upscalée BGR uint8, ou l'originale si ESRGAN échoue.
        """
        if not self._load_session(log_fn):
            return _bicubic_upscale(img_bgr, ESRGAN_SCALE)

        try:
            # Normalisation RGB float32 [0, 1] — format attendu par le modèle ONNX
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            # NCHW : (1, 3, H, W)
            inp = np.transpose(img_rgb, (2, 0, 1))[np.newaxis, ...]

            # Inférence ONNX
            out = self._session.run(
                [self._output_name], {self._input_name: inp}
            )[0]

            # NCHW → HWC, clip, uint8
            out = np.transpose(out[0], (1, 2, 0))
            out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
            result = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

            log_fn(f"[ESRGAN] {img_bgr.shape[1]}×{img_bgr.shape[0]}px "
                   f"→ {result.shape[1]}×{result.shape[0]}px (×{ESRGAN_SCALE})")
            return result

        except Exception as e:
            log_fn(f"[ESRGAN] Erreur inférence : {e} — fallback bicubique")
            return _bicubic_upscale(img_bgr, ESRGAN_SCALE)


# Singleton ESRGAN
_ESRGAN_INSTANCE: Optional[ESRGANUpscaler] = None

def get_esrgan() -> ESRGANUpscaler:
    global _ESRGAN_INSTANCE
    if _ESRGAN_INSTANCE is None:
        _ESRGAN_INSTANCE = ESRGANUpscaler()
    return _ESRGAN_INSTANCE


# ── Fallback bicubique ────────────────────────────────────────────────────────
def _bicubic_upscale(img: np.ndarray, scale: int) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


# ── Lecture image ─────────────────────────────────────────────────────────────
def _load_bgr(image: Union[str, Path, bytes, io.BytesIO]) -> Optional[np.ndarray]:
    if isinstance(image, (str, Path)):
        return cv2.imread(str(image))
    raw = image.read() if hasattr(image, "read") else image
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(
    image: Union[str, Path, bytes, io.BytesIO],
    log_fn: Optional[Callable] = None,
    use_esrgan: bool = True,
    stealth: bool = False,
) -> io.BytesIO:
    """
    Pipeline complet v2 : ESRGAN → CLAHE → Denoise → Sharpen.

    Args:
        image      : chemin, bytes ou BytesIO
        log_fn     : callable de log (print par défaut)
        use_esrgan : activer Real-ESRGAN sur petites images
        stealth    : mode furtif (pas d'écriture disque — déjà géré en amont)

    Retourne un BytesIO JPEG qualité 95 prêt pour InsightFace + MediaPipe.
    """

    def _log(msg: str):
        if log_fn:
            try:
                log_fn(msg)
            except Exception:
                pass
        else:
            print(msg)

    # ── Lecture ──────────────────────────────────────────────────────────────
    img = _load_bgr(image)
    if img is None:
        _log("[PREPROC] Image illisible — retour image originale")
        return _fallback_return(image)

    h, w = img.shape[:2]
    _log(f"[PREPROC] Entrée : {w}×{h}px")

    # Cap résolution max — InsightFace det_size=(640,640) optimal sous 1280px [FIX]
    _MAX_DIM = 1280
    if max(h, w) > _MAX_DIM:
        scale_cap = _MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale_cap), int(h * scale_cap)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        _log(f"[PREPROC] Résolution réduite → {w}×{h}px (InsightFace cap)")

    # ── Étape 1 : Super-Résolution ESRGAN ────────────────────────────────────
    if use_esrgan and min(h, w) < ESRGAN_MIN_PX:
        _log(f"[PREPROC] Image < {ESRGAN_MIN_PX}px → activation ESRGAN ×{ESRGAN_SCALE}")
        img = get_esrgan().upscale(img, _log)
        h, w = img.shape[:2]

    # ── Étape 1b : Upscale bicubique (images entre 128 et 200px) ────────────
    elif min(h, w) < BICUBIC_MIN_PX:
        scale = BICUBIC_MIN_PX / min(h, w)
        img   = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        h, w  = img.shape[:2]
        _log(f"[PREPROC] Upscale bicubique ×{scale:.1f} → {w}×{h}px")

    # ── Étape 2 : CLAHE (espace LAB, canal L) ────────────────────────────────
    lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq  = clahe.apply(l)
    img   = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)
    _log("[PREPROC] CLAHE appliqué")

    # ── Étape 3 : Denoising ──────────────────────────────────────────────────
    img = cv2.fastNlMeansDenoisingColored(
        img, None, h=7, hColor=7, templateWindowSize=7, searchWindowSize=21
    )
    _log("[PREPROC] Denoising NlMeans appliqué")

    # ── Étape 4 : Sharpening (Unsharp Masking) ───────────────────────────────
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
    img  = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    img  = np.clip(img, 0, 255).astype(np.uint8)
    _log("[PREPROC] Sharpening appliqué")

    # ── Export ────────────────────────────────────────────────────────────────
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        _log("[PREPROC] Encodage JPEG échoué — retour image originale")
        return _fallback_return(image)

    _log(f"[PREPROC] Sortie : {img.shape[1]}×{img.shape[0]}px — pipeline terminé ✓")
    return io.BytesIO(buf.tobytes())


def _fallback_return(image) -> io.BytesIO:
    if isinstance(image, io.BytesIO):
        image.seek(0)
        return image
    if isinstance(image, (str, Path)):
        with open(str(image), "rb") as f:
            return io.BytesIO(f.read())
    return io.BytesIO(image if isinstance(image, bytes) else bytes(image))

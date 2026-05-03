"""
api/main.py  —  OMNI-RECO v2.0
Dashboard FastAPI + REST API embarquable

Routes :
  POST /compare              → comparaison de deux images (upload)
  POST /compare/urls         → comparaison via URLs
  POST /analyze              → fiche biométrique complète d'une image
  GET  /health               → statut du moteur
  GET  /docs                 → Swagger UI (auto-généré par FastAPI)
  GET  /redoc                → ReDoc

Dashboard visuel :
  GET  /                     → interface web interactive (upload + résultat)

Lancement :
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Intégration dans un autre logiciel :
  import httpx
  r = httpx.post("http://localhost:8000/compare/urls",
                 json={"url_a": "...", "url_b": "..."})
  print(r.json())
"""

import io
import base64
import time
import asyncio
from pathlib import Path
from typing  import Optional

from fastapi              import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses    import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic             import BaseModel, HttpUrl

# ── Import moteur ─────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.face_engine import FaceEngine

# ══════════════════════════════════════════════════════════════════════════════
#  APP FASTAPI
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = "OMNI-RECO v2.0 API",
    description = "Moteur de reconnaissance faciale chirurgical — ArcFace + Géométrie 3D + Texture Gabor + Oreilles",
    version     = "2.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# Singleton moteur (chargé au premier appel)
_ENGINE: Optional[FaceEngine] = None
_ENGINE_LOCK = asyncio.Lock()

async def get_engine() -> FaceEngine:
    global _ENGINE
    if _ENGINE is None:
        async with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = FaceEngine()
    return _ENGINE


# ══════════════════════════════════════════════════════════════════════════════
#  MODÈLES PYDANTIC (Schémas de requête / réponse)
# ══════════════════════════════════════════════════════════════════════════════

class CompareUrlsRequest(BaseModel):
    url_a:   str
    url_b:   str
    verbose: bool = False

class CompareResponse(BaseModel):
    match:      bool
    score:      float
    confidence: float
    method:     str
    detail:     dict
    elapsed_s:  float

class AnalyzeResponse(BaseModel):
    ok:         bool
    quality:    dict
    severity:   str
    is_partial: bool
    yaw:        float
    geo_ok:     bool
    tex_ok:     bool
    tex_method: str
    elapsed_s:  float


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES API
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    """Statut du service."""
    return {"status": "ok", "version": "2.0.0", "engine": "ready"}


@app.post("/compare", response_model=CompareResponse, summary="Comparer deux images (upload)")
async def compare_upload(
    image_a: UploadFile = File(..., description="Image source"),
    image_b: UploadFile = File(..., description="Image cible"),
):
    """
    Compare deux images faciales uploadées directement.
    Retourne le score de fusion et le verdict (match/no match).
    """
    engine = await get_engine()
    bytes_a = await image_a.read()
    bytes_b = await image_b.read()

    if not bytes_a or not bytes_b:
        raise HTTPException(400, "Images vides")

    t0     = time.perf_counter()
    result = await engine.process_pair(bytes_a, bytes_b)
    elapsed = round(time.perf_counter() - t0, 3)

    if "error" in result:
        raise HTTPException(422, result["error"])

    return {**result, "elapsed_s": elapsed}


@app.post("/compare/urls", response_model=CompareResponse, summary="Comparer via URLs")
async def compare_urls(req: CompareUrlsRequest):
    """
    Compare deux images depuis leurs URLs.
    Utile pour intégration dans des workflows automatisés.
    """
    import urllib.request

    engine = await get_engine()

    def _fetch(url: str) -> bytes:
        with urllib.request.urlopen(url, timeout=15) as r:
            return r.read()

    loop = asyncio.get_event_loop()
    try:
        bytes_a = await loop.run_in_executor(None, _fetch, req.url_a)
        bytes_b = await loop.run_in_executor(None, _fetch, req.url_b)
    except Exception as e:
        raise HTTPException(400, f"Téléchargement échoué : {e}")

    t0     = time.perf_counter()
    result = await engine.process_pair(bytes_a, bytes_b)
    elapsed = round(time.perf_counter() - t0, 3)

    if "error" in result:
        raise HTTPException(422, result["error"])

    return {**result, "elapsed_s": elapsed}


@app.post("/analyze", response_model=AnalyzeResponse, summary="Analyser une image")
async def analyze_image(image: UploadFile = File(...)):
    """
    Analyse complète d'une image : qualité, pose, sévérité, géométrie, texture.
    Ne compare pas — génère la fiche biométrique complète.
    """
    engine  = await get_engine()
    content = await image.read()
    if not content:
        raise HTTPException(400, "Image vide")

    t0    = time.perf_counter()
    fiche = await engine.process_single(content)
    elapsed = round(time.perf_counter() - t0, 3)

    return {
        "ok":         fiche.get("ok", False),
        "quality":    fiche.get("quality", {}),
        "severity":   fiche.get("severity", "?"),
        "is_partial": fiche.get("is_partial", False),
        "yaw":        fiche.get("yaw", 0.0),
        "geo_ok":     fiche.get("geo_sig", {}).get("ok", False),
        "tex_ok":     fiche.get("tex_sig", {}).get("ok", False),
        "tex_method": fiche.get("tex_sig", {}).get("method", "?"),
        "elapsed_s":  elapsed,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD HTML
# ══════════════════════════════════════════════════════════════════════════════

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OMNI-RECO v2.0 — Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d0d1a; color: #e0e0e0; font-family: 'Segoe UI', monospace; }
  header { padding: 24px 40px; border-bottom: 1px solid #1a1a3a;
           display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 1.5rem; color: #4fc3f7; letter-spacing: 2px; }
  header span { font-size: 0.8rem; color: #555; }
  .container { max-width: 960px; margin: 40px auto; padding: 0 24px; }
  .card { background: #0f0f22; border: 1px solid #1e1e3e;
          border-radius: 12px; padding: 28px; margin-bottom: 24px; }
  .card h2 { color: #4fc3f7; font-size: 1rem; margin-bottom: 20px;
             text-transform: uppercase; letter-spacing: 1px; }
  .upload-zone { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
  .drop-target { border: 2px dashed #2a2a4a; border-radius: 8px; padding: 24px;
                 text-align: center; cursor: pointer; transition: border-color 0.2s;
                 background: #08080f; }
  .drop-target:hover, .drop-target.active { border-color: #4fc3f7; }
  .drop-target img { max-width: 100%; max-height: 180px; border-radius: 6px; margin-top: 10px; }
  .drop-target input[type=file] { display: none; }
  .drop-target label { cursor: pointer; color: #4fc3f7; font-size: 0.85rem; }
  .btn { background: #1a3a5c; color: #4fc3f7; border: 1px solid #4fc3f7;
         padding: 12px 32px; border-radius: 8px; cursor: pointer;
         font-size: 0.95rem; letter-spacing: 1px; transition: all 0.2s; }
  .btn:hover { background: #4fc3f7; color: #0d0d1a; }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .result-box { display: none; }
  .verdict { font-size: 2rem; font-weight: bold; text-align: center;
             padding: 20px; border-radius: 8px; margin: 16px 0; }
  .verdict.match    { background: #0a2a0a; color: #4caf50; border: 1px solid #4caf50; }
  .verdict.no-match { background: #2a0a0a; color: #f44336; border: 1px solid #f44336; }
  .scores-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 16px 0; }
  .score-card  { background: #08080f; border: 1px solid #1e1e3e;
                 border-radius: 8px; padding: 14px; text-align: center; }
  .score-card .value { font-size: 1.4rem; font-weight: bold; color: #4fc3f7; }
  .score-card .label { font-size: 0.75rem; color: #666; margin-top: 4px; }
  .detail-row { display: flex; justify-content: space-between;
                padding: 8px 0; border-bottom: 1px solid #111130; font-size: 0.85rem; }
  .detail-row .key { color: #888; }
  .detail-row .val { color: #ccc; font-family: monospace; }
  .badge { padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; }
  .badge.high   { background: #0a2a0a; color: #4caf50; }
  .badge.medium { background: #2a2a0a; color: #ffb300; }
  .badge.low    { background: #2a0a0a; color: #f44336; }
  .loader { display: none; text-align: center; padding: 20px; color: #4fc3f7; }
  .api-block { background: #06060f; border: 1px solid #1a1a2e;
               border-radius: 8px; padding: 16px; margin-top: 16px; }
  .api-block pre { color: #80cbc4; font-size: 0.8rem; overflow-x: auto; }
  .tabs { display: flex; gap: 8px; margin-bottom: 16px; }
  .tab-btn { background: none; border: 1px solid #2a2a4a; color: #888;
             padding: 6px 16px; border-radius: 6px; cursor: pointer; font-size: 0.8rem; }
  .tab-btn.active { border-color: #4fc3f7; color: #4fc3f7; }
  .tab-content { display: none; }
  .tab-content.active { display: block; }
  footer { text-align: center; padding: 32px; color: #2a2a4a; font-size: 0.75rem; }
</style>
</head>
<body>

<header>
  <div>
    <h1>⬡ OMNI-RECO v2.0</h1>
    <span>ArcFace 512D · MediaPipe 3D · Gabor Texture · Ear Biometrics</span>
  </div>
</header>

<div class="container">

  <!-- TABS -->
  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('compare')">Comparer</button>
    <button class="tab-btn"        onclick="switchTab('analyze')">Analyser</button>
    <button class="tab-btn"        onclick="switchTab('api')">API REST</button>
  </div>

  <!-- TAB COMPARER -->
  <div id="tab-compare" class="tab-content active">
    <div class="card">
      <h2>Comparaison de deux visages</h2>
      <div class="upload-zone">
        <div class="drop-target" id="zone-a" onclick="document.getElementById('file-a').click()">
          <label>Image A — cliquer ou glisser</label>
          <input type="file" id="file-a" accept="image/*" onchange="previewImage(this,'zone-a','prev-a')">
          <img id="prev-a" style="display:none">
        </div>
        <div class="drop-target" id="zone-b" onclick="document.getElementById('file-b').click()">
          <label>Image B — cliquer ou glisser</label>
          <input type="file" id="file-b" accept="image/*" onchange="previewImage(this,'zone-b','prev-b')">
          <img id="prev-b" style="display:none">
        </div>
      </div>
      <button class="btn" id="btn-compare" onclick="doCompare()">⬡ ANALYSER</button>

      <div class="loader" id="loader-compare">⬡ Traitement en cours…</div>

      <div class="result-box" id="result-compare">
        <div class="verdict" id="verdict-text"></div>
        <div class="scores-grid">
          <div class="score-card">
            <div class="value" id="sc-fused">—</div>
            <div class="label">Score Fusionné</div>
          </div>
          <div class="score-card">
            <div class="value" id="sc-arcface">—</div>
            <div class="label">ArcFace</div>
          </div>
          <div class="score-card">
            <div class="value" id="sc-geo">—</div>
            <div class="label">Géométrie 3D</div>
          </div>
        </div>
        <div id="detail-rows"></div>
      </div>
    </div>
  </div>

  <!-- TAB ANALYSER -->
  <div id="tab-analyze" class="tab-content">
    <div class="card">
      <h2>Analyse biométrique d'une image</h2>
      <div class="drop-target" style="width:50%;min-width:200px"
           onclick="document.getElementById('file-analyze').click()">
        <label>Image à analyser</label>
        <input type="file" id="file-analyze" accept="image/*"
               onchange="previewImage(this,'drop-analyze','prev-analyze')">
        <img id="prev-analyze" style="display:none">
      </div>
      <br>
      <button class="btn" id="btn-analyze" onclick="doAnalyze()">⬡ ANALYSER</button>
      <div class="loader"     id="loader-analyze">⬡ Analyse en cours…</div>
      <div class="result-box" id="result-analyze">
        <div id="analyze-rows"></div>
      </div>
    </div>
  </div>

  <!-- TAB API -->
  <div id="tab-api" class="tab-content">
    <div class="card">
      <h2>API REST — Intégration</h2>
      <div class="api-block">
        <pre>
# Comparaison par upload
curl -X POST http://localhost:8000/compare \\
     -F "image_a=@photo_a.jpg" \\
     -F "image_b=@photo_b.jpg"

# Comparaison par URLs
curl -X POST http://localhost:8000/compare/urls \\
     -H "Content-Type: application/json" \\
     -d '{"url_a":"https://...","url_b":"https://..."}'

# Analyse biométrique
curl -X POST http://localhost:8000/analyze \\
     -F "image=@photo.jpg"

# Réponse type
{
  "match": true,
  "score": 0.82,
  "confidence": 82.0,
  "method": "arcface+geometry+texture",
  "detail": {
    "arcface_sim": 0.88,
    "geo_score": 0.74,
    "texture_score": 0.71,
    "severity": "high",
    "partial_a": false,
    "partial_b": false,
    "weights": {"arcface": 0.65, "geometry": 0.25, "texture": 0.10}
  },
  "elapsed_s": 1.24
}

# Python httpx
import httpx
r = httpx.post("http://localhost:8000/compare/urls",
               json={"url_a": "...", "url_b": "..."})
print(r.json())

# Swagger interactif → http://localhost:8000/docs
        </pre>
      </div>
    </div>
  </div>

</div>

<footer>OMNI-RECO v2.0 — Usage privé — Recherche & Développement</footer>

<script>
function switchTab(name) {
  document.querySelectorAll('.tab-btn').forEach((b,i)=>b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c=>c.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('tab-'+name).classList.add('active');
}

function previewImage(input, zoneId, prevId) {
  const file = input.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    const img = document.getElementById(prevId);
    img.src = e.target.result;
    img.style.display = 'block';
  };
  reader.readAsDataURL(file);
}

async function doCompare() {
  const fa = document.getElementById('file-a').files[0];
  const fb = document.getElementById('file-b').files[0];
  if (!fa || !fb) { alert('Sélectionner deux images'); return; }

  document.getElementById('loader-compare').style.display = 'block';
  document.getElementById('result-compare').style.display = 'none';
  document.getElementById('btn-compare').disabled = true;

  const fd = new FormData();
  fd.append('image_a', fa);
  fd.append('image_b', fb);

  try {
    const r = await fetch('/compare', {method:'POST', body: fd});
    if (!r.ok) { const e = await r.json(); throw new Error(e.detail); }
    const d = await r.json();
    renderCompareResult(d);
  } catch(e) {
    alert('Erreur : ' + e.message);
  } finally {
    document.getElementById('loader-compare').style.display = 'none';
    document.getElementById('btn-compare').disabled = false;
  }
}

function renderCompareResult(d) {
  const box = document.getElementById('result-compare');
  box.style.display = 'block';

  const v = document.getElementById('verdict-text');
  if (d.match) { v.textContent = '✅  MATCH — même personne'; v.className = 'verdict match'; }
  else          { v.textContent = '❌  NO MATCH — personnes différentes'; v.className = 'verdict no-match'; }

  document.getElementById('sc-fused').textContent   = d.score.toFixed(4);
  document.getElementById('sc-arcface').textContent  = (d.detail.arcface_sim||'—').toFixed ? d.detail.arcface_sim.toFixed(4) : '—';
  document.getElementById('sc-geo').textContent      = d.detail.geo_score != null ? d.detail.geo_score.toFixed(4) : '—';

  const rows = document.getElementById('detail-rows');
  const sev  = d.detail.severity || 'medium';
  rows.innerHTML = `
    <div class="detail-row"><span class="key">Confiance</span><span class="val">${d.confidence.toFixed(1)}%</span></div>
    <div class="detail-row"><span class="key">Texture Gabor</span><span class="val">${d.detail.texture_score != null ? d.detail.texture_score.toFixed(4) : '—'}</span></div>
    <div class="detail-row"><span class="key">Sévérité qualité</span><span class="val"><span class="badge ${sev}">${sev.toUpperCase()}</span></span></div>
    <div class="detail-row"><span class="key">Méthode</span><span class="val">${d.method}</span></div>
    <div class="detail-row"><span class="key">Visage partiel A/B</span><span class="val">${d.detail.partial_a?'⚠ oui':'non'} / ${d.detail.partial_b?'⚠ oui':'non'}</span></div>
    <div class="detail-row"><span class="key">Temps traitement</span><span class="val">${d.elapsed_s}s</span></div>
    <div class="detail-row"><span class="key">Poids (A/G/T)</span><span class="val">${d.detail.weights ? Object.values(d.detail.weights).map(v=>(v*100).toFixed(0)+'%').join(' / ') : '—'}</span></div>
  `;
}

async function doAnalyze() {
  const f = document.getElementById('file-analyze').files[0];
  if (!f) { alert('Sélectionner une image'); return; }

  document.getElementById('loader-analyze').style.display = 'block';
  document.getElementById('result-analyze').style.display = 'none';
  document.getElementById('btn-analyze').disabled = true;

  const fd = new FormData();
  fd.append('image', f);

  try {
    const r = await fetch('/analyze', {method:'POST', body: fd});
    if (!r.ok) { const e = await r.json(); throw new Error(e.detail); }
    const d = await r.json();
    renderAnalyzeResult(d);
  } catch(e) {
    alert('Erreur : ' + e.message);
  } finally {
    document.getElementById('loader-analyze').style.display = 'none';
    document.getElementById('btn-analyze').disabled = false;
  }
}

function renderAnalyzeResult(d) {
  const box  = document.getElementById('result-analyze');
  box.style.display = 'block';
  const sev  = d.severity || 'medium';
  const q    = d.quality  || {};
  document.getElementById('analyze-rows').innerHTML = `
    <div class="detail-row"><span class="key">Statut</span><span class="val">${d.ok ? '✅ OK' : '❌ Échec'}</span></div>
    <div class="detail-row"><span class="key">Sévérité</span><span class="val"><span class="badge ${sev}">${sev.toUpperCase()}</span></span></div>
    <div class="detail-row"><span class="key">Score qualité</span><span class="val">${(q.global_score||0).toFixed(3)}</span></div>
    <div class="detail-row"><span class="key">Visage partiel</span><span class="val">${d.is_partial ? '⚠ oui' : 'non'}</span></div>
    <div class="detail-row"><span class="key">Yaw estimé</span><span class="val">${d.yaw.toFixed(1)}°</span></div>
    <div class="detail-row"><span class="key">Géométrie 3D</span><span class="val">${d.geo_ok ? '✅ OK' : '❌'}</span></div>
    <div class="detail-row"><span class="key">Texture</span><span class="val">${d.tex_ok ? '✅ ' + d.tex_method : '❌'}</span></div>
    <div class="detail-row"><span class="key">Temps</span><span class="val">${d.elapsed_s}s</span></div>
  `;
}

// Drag & Drop
['zone-a','zone-b'].forEach(id => {
  const el = document.getElementById(id);
  el.addEventListener('dragover',  e => { e.preventDefault(); el.classList.add('active'); });
  el.addEventListener('dragleave', () => el.classList.remove('active'));
  el.addEventListener('drop', e => {
    e.preventDefault();
    el.classList.remove('active');
    const file = e.dataTransfer.files[0];
    if (!file) return;
    const inputId = id === 'zone-a' ? 'file-a' : 'file-b';
    const prevId  = id === 'zone-a' ? 'prev-a' : 'prev-b';
    const dt = new DataTransfer();
    dt.items.add(file);
    document.getElementById(inputId).files = dt.files;
    previewImage({files:[file]}, id, prevId);
  });
});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Dashboard visuel interactif."""
    return DASHBOARD_HTML

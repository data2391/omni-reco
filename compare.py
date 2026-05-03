"""
compare.py  —  OMNI-RECO v2.0
CLI de comparaison de deux images faciales.

Usage :
  python compare.py <image_a> <image_b> [--verbose] [--model buffalo_l]

Exemples :
  python compare.py photo1.jpg photo2.png
  python compare.py https://example.com/a.jpg local_b.jpg --verbose
  python compare.py a.jpg b.jpg --model buffalo_s
"""

import sys
import argparse
import asyncio
import io
import time
from pathlib import Path
from typing import Union

# ── Tentative import Rich pour un beau terminal ───────────────────────────────
try:
    from rich.console import Console
    from rich.table   import Table
    from rich.panel   import Panel
    from rich         import print as rprint
    HAS_RICH = True
    console  = Console()
except ImportError:
    HAS_RICH = False
    console  = None


def _load_input(src: str) -> Union[bytes, Path]:
    """Charge depuis URL ou chemin local."""
    if src.startswith("http://") or src.startswith("https://"):
        try:
            import urllib.request
            with urllib.request.urlopen(src, timeout=15) as r:
                return r.read()
        except Exception as e:
            raise SystemExit(f"[ERREUR] Téléchargement impossible : {src} → {e}")
    p = Path(src)
    if not p.exists():
        raise SystemExit(f"[ERREUR] Fichier introuvable : {src}")
    return p


def _print_verdict(result: dict, verbose: bool):
    match      = result.get("match", False)
    score      = result.get("score", 0.0)
    confidence = result.get("confidence", 0.0)
    detail     = result.get("detail", {})
    method     = result.get("method", "?")

    match_emoji = "✅  MATCH" if match else "❌  NO MATCH"
    match_color = "green" if match else "red"

    if HAS_RICH:
        # Tableau principal
        t = Table(show_header=True, header_style="bold cyan")
        t.add_column("Composante",    style="cyan", width=22)
        t.add_column("Valeur",        justify="right")
        t.add_column("Détail",        style="dim")

        t.add_row("Score fusionné",   f"[bold]{score:.4f}[/bold]",    f"Méthode : {method}")
        t.add_row("Confiance",        f"{confidence:.1f}%",            "")
        t.add_row("Sévérité qualité", detail.get("severity","?"),      "")
        t.add_row("ArcFace sim",      f"{detail.get('arcface_sim','-'):.4f}" if detail.get('arcface_sim') is not None else "-", "dist cosinus inversée")
        if detail.get("geo_score") is not None:
            t.add_row("Géométrie 3D",  f"{detail['geo_score']:.4f}",   "MediaPipe 18 ratios")
        if detail.get("texture_score") is not None:
            t.add_row("Texture Gabor", f"{detail['texture_score']:.4f}","40 filtres × zones peau")

        if detail.get("partial_a") or detail.get("partial_b"):
            t.add_row("⚠ Visage partiel",
                f"A={detail.get('yaw_a',0):.1f}° B={detail.get('yaw_b',0):.1f}°",
                "pondération adaptée")

        poids = detail.get("weights", {})
        if poids:
            t.add_row("Poids (A/G/T)",
                f"{poids.get('arcface',0)*100:.0f}% / {poids.get('geometry',0)*100:.0f}% / {poids.get('texture',0)*100:.0f}%",
                "adaptatif selon qualité")

        console.print(Panel(f"[bold {match_color}]{match_emoji}[/bold {match_color}]",
                            border_style=match_color))
        console.print(t)

        if verbose:
            console.rule("[dim]Logs")
            for line in result.get("log_a", []):
                console.print(f"  [dim][A][/dim] {line}")
            for line in result.get("log_b", []):
                console.print(f"  [dim][B][/dim] {line}")
    else:
        print("=" * 50)
        print(f"  {match_emoji}")
        print(f"  Score    : {score:.4f}")
        print(f"  Confiance: {confidence:.1f}%")
        print(f"  Méthode  : {method}")
        if detail.get("arcface_sim") is not None:
            print(f"  ArcFace  : {detail['arcface_sim']:.4f}")
        if detail.get("geo_score") is not None:
            print(f"  Géométrie: {detail['geo_score']:.4f}")
        if detail.get("texture_score") is not None:
            print(f"  Texture  : {detail['texture_score']:.4f}")
        print("=" * 50)


async def _main(args):
    # Import ici pour ne pas crasher si InsightFace absent (affiche erreur propre)
    try:
        from core.face_engine import FaceEngine
    except ImportError as e:
        raise SystemExit(f"[ERREUR] Import FaceEngine échoué : {e}\n"
                         "Vérifie que insightface, mediapipe, opencv sont installés.")

    log_fn = (lambda m: console.print(f"  [dim]{m}[/dim]")) if (args.verbose and HAS_RICH) else (
             (lambda m: print(f"  {m}")) if args.verbose else None)

    engine = FaceEngine(model_name=args.model, log_fn=log_fn)

    src_a = _load_input(args.image_a)
    src_b = _load_input(args.image_b)

    if HAS_RICH:
        console.rule("[bold cyan]OMNI-RECO v2.0[/bold cyan]")

    t0     = time.perf_counter()
    result = await engine.process_pair(src_a, src_b)
    elapsed = time.perf_counter() - t0

    _print_verdict(result, args.verbose)

    if HAS_RICH:
        console.print(f"[dim]  Temps total : {elapsed:.2f}s[/dim]")
    else:
        print(f"  Temps : {elapsed:.2f}s")

    return 0 if result.get("match") else 1


def main():
    parser = argparse.ArgumentParser(
        description="OMNI-RECO v2.0 — Comparaison faciale chirurgicale"
    )
    parser.add_argument("image_a",          help="Image source (chemin ou URL)")
    parser.add_argument("image_b",          help="Image cible (chemin ou URL)")
    parser.add_argument("--verbose", "-v",  action="store_true")
    parser.add_argument("--model",   "-m",  default="buffalo_l",
                        choices=["buffalo_l", "buffalo_s"],
                        help="Modèle InsightFace (défaut: buffalo_l)")
    args = parser.parse_args()
    sys.exit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()

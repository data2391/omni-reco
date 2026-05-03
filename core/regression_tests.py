"""
core/regression_tests.py  —  OMNI-RECO v2.0
Suite de tests de régression + calibration des seuils

Objectif : tester le moteur sur des paires connues (sosies, mêmes personnes,
personnes différentes) et trouver les seuils optimaux par composante.

Paires de test intégrées (URLs publiques, domaine libre) :
  - Sosies vrais  : paires visuellement proches mais personnes DIFFÉRENTES
  - Même personne : plusieurs photos de la même personne
  - Négatifs clairs : personnes clairement différentes

Usage :
  python -m core.regression_tests
  python -m core.regression_tests --detailed
  python -m core.regression_tests --calibrate   # recalcule les seuils optimaux
"""

import asyncio
import argparse
import json
import time
import urllib.request
import io
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

# ── Import conditionnel (scikit-learn optionnel pour calibration) ─────────────
try:
    from sklearn.metrics import roc_curve, roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from rich.console import Console
    from rich.table   import Table
    from rich         import print as rprint
    HAS_RICH = True
    console  = Console()
except ImportError:
    HAS_RICH = False
    console  = None


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET DE TEST INTÉGRÉ
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestPair:
    id:          str
    label:       str      # "same" | "different" | "lookalike"
    expected:    bool     # True = match attendu
    desc:        str
    url_a:       str
    url_b:       str
    # Résultat rempli après test
    result:      Optional[dict] = field(default=None, repr=False)
    error:       Optional[str]  = None
    elapsed_s:   float          = 0.0


# Paires de test avec images Wikimedia Commons (domaine public / CC)
TEST_PAIRS: list = [
    # ── Même personne, conditions différentes ────────────────────────────────
    TestPair(
        id="tp_same_01", label="same", expected=True,
        desc="Même personne — éclairage différent (test cohérence ArcFace)",
        url_a="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg",
        url_b="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg",
    ),
    # ── Personnes différentes ────────────────────────────────────────────────
    TestPair(
        id="tp_diff_01", label="different", expected=False,
        desc="Personnes différentes — test négatif clair",
        url_a="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg",
        url_b="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/320px-Cat_November_2010-1a.jpg",
    ),
]
# NOTE : dans un usage réel, remplacer les URLs par un dataset local
# structuré comme : data/pairs/same/p001_a.jpg, data/pairs/same/p001_b.jpg


# ══════════════════════════════════════════════════════════════════════════════
#  RUNNER
# ══════════════════════════════════════════════════════════════════════════════

async def _run_pair(pair: TestPair, engine) -> TestPair:
    """Exécute un test sur une paire."""
    try:
        # Téléchargement
        def _fetch(url: str) -> bytes:
            with urllib.request.urlopen(url, timeout=15) as r:
                return r.read()

        loop    = asyncio.get_event_loop()
        bytes_a = await loop.run_in_executor(None, _fetch, pair.url_a)
        bytes_b = await loop.run_in_executor(None, _fetch, pair.url_b)

        t0 = time.perf_counter()
        result = await engine.process_pair(bytes_a, bytes_b)
        pair.elapsed_s = round(time.perf_counter() - t0, 2)
        pair.result    = result

    except Exception as e:
        pair.error = str(e)

    return pair


async def run_all(
    pairs: list,
    model_name: str = "buffalo_l",
    log_fn=None,
) -> list:
    """Lance tous les tests en parallèle par batch de 3."""
    from core.face_engine import FaceEngine
    engine = FaceEngine(model_name=model_name, log_fn=log_fn)

    results = []
    batch_size = 3
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        done  = await asyncio.gather(*[_run_pair(p, engine) for p in batch])
        results.extend(done)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSE DES RÉSULTATS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RegressionReport:
    n_pairs:       int   = 0
    n_correct:     int   = 0
    n_errors:      int   = 0
    accuracy:      float = 0.0
    # Précision par catégorie
    same_tp:       int   = 0   # vrais positifs
    same_fn:       int   = 0   # faux négatifs (raté un match)
    diff_tn:       int   = 0   # vrais négatifs
    diff_fp:       int   = 0   # faux positifs (confondu deux persos)
    look_tp:       int   = 0   # sosie détecté comme différent (=correct)
    look_fp:       int   = 0   # sosie confondu avec même personne
    # Statistiques de score
    scores_same:   list  = field(default_factory=list)
    scores_diff:   list  = field(default_factory=list)
    scores_look:   list  = field(default_factory=list)
    # Timing
    avg_time_s:    float = 0.0
    # Seuils calibrés
    calibrated_threshold: Optional[float] = None
    auc:                  Optional[float] = None


def analyze(pairs: list) -> RegressionReport:
    rep = RegressionReport(n_pairs=len(pairs))

    all_scores = []
    all_labels = []
    times      = []

    for p in pairs:
        if p.error:
            rep.n_errors += 1
            continue
        if not p.result:
            rep.n_errors += 1
            continue

        score   = p.result.get("score", 0.0)
        matched = p.result.get("match", False)
        correct = (matched == p.expected)

        if correct:
            rep.n_correct += 1

        times.append(p.elapsed_s)
        all_scores.append(score)
        all_labels.append(1 if p.expected else 0)

        if p.label == "same":
            rep.scores_same.append(score)
            if matched:  rep.same_tp += 1
            else:        rep.same_fn += 1
        elif p.label == "different":
            rep.scores_diff.append(score)
            if not matched: rep.diff_tn += 1
            else:           rep.diff_fp += 1
        elif p.label == "lookalike":
            rep.scores_look.append(score)
            if not matched: rep.look_tp += 1
            else:           rep.look_fp += 1

    n_valid = rep.n_pairs - rep.n_errors
    rep.accuracy  = round(rep.n_correct / n_valid, 4) if n_valid else 0.0
    rep.avg_time_s = round(sum(times) / len(times), 2) if times else 0.0

    # Calibration automatique via ROC (si scikit-learn disponible)
    if HAS_SKLEARN and len(set(all_labels)) == 2:
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        rep.auc = round(roc_auc_score(all_labels, all_scores), 4)
        # Seuil optimal : maximise TPR - FPR (critère de Youden)
        youden_idx = (tpr - fpr).argmax()
        rep.calibrated_threshold = round(float(thresholds[youden_idx]), 4)

    return rep


# ══════════════════════════════════════════════════════════════════════════════
#  AFFICHAGE
# ══════════════════════════════════════════════════════════════════════════════

def print_report(pairs: list, rep: RegressionReport, detailed: bool = False):
    n_valid = rep.n_pairs - rep.n_errors

    if HAS_RICH:
        console.rule("[bold cyan]OMNI-RECO v2.0 — Rapport de Régression[/bold cyan]")

        # Tableau résumé
        t = Table(show_header=True, header_style="bold cyan")
        t.add_column("Métrique",       width=28)
        t.add_column("Valeur",         justify="right")
        t.add_column("Détail",         style="dim")

        t.add_row("Paires testées",     str(n_valid),          f"{rep.n_errors} erreurs")
        acc_color = "green" if rep.accuracy >= 0.9 else ("yellow" if rep.accuracy >= 0.75 else "red")
        t.add_row("Accuracy globale",   f"[{acc_color}]{rep.accuracy*100:.1f}%[/{acc_color}]", "")
        t.add_row("True Positives",     str(rep.same_tp),       "même personne détectée")
        t.add_row("False Negatives",    str(rep.same_fn),       "match raté")
        t.add_row("True Negatives",     str(rep.diff_tn),       "différence détectée")
        t.add_row("False Positives",    str(rep.diff_fp),       "confusion identité")
        if rep.look_tp or rep.look_fp:
            t.add_row("Sosies (TN)",    str(rep.look_tp),       "sosie correctement rejeté")
            t.add_row("Sosies (FP)",    str(rep.look_fp),       "sosie confondu ⚠")
        t.add_row("Temps moyen",        f"{rep.avg_time_s}s",   "par paire")
        if rep.auc:
            t.add_row("AUC ROC",        f"{rep.auc:.4f}",       "")
        if rep.calibrated_threshold:
            t.add_row("Seuil calibré",  f"{rep.calibrated_threshold:.4f}",
                      "→ Youden optimal")
        console.print(t)

        # Distributions de scores
        if rep.scores_same:
            mean_s = sum(rep.scores_same) / len(rep.scores_same)
            console.print(f"  Score moyen [same]     : [green]{mean_s:.4f}[/green]")
        if rep.scores_diff:
            mean_d = sum(rep.scores_diff) / len(rep.scores_diff)
            console.print(f"  Score moyen [different]: [red]{mean_d:.4f}[/red]")
        if rep.scores_look:
            mean_l = sum(rep.scores_look) / len(rep.scores_look)
            console.print(f"  Score moyen [lookalike]: [yellow]{mean_l:.4f}[/yellow]")

        # Détail par paire
        if detailed:
            console.rule("[dim]Détail paire par paire[/dim]")
            dt = Table(show_header=True, header_style="dim")
            dt.add_column("ID",      width=15)
            dt.add_column("Label",   width=12)
            dt.add_column("Score",   justify="right")
            dt.add_column("Match",   justify="center")
            dt.add_column("Correct", justify="center")
            dt.add_column("Temps")
            dt.add_column("Méthode")
            for p in pairs:
                if p.error:
                    dt.add_row(p.id, p.label, "ERR", "ERR", "❌", "-", p.error[:30])
                    continue
                if not p.result:
                    continue
                score   = p.result.get("score", 0.0)
                matched = p.result.get("match", False)
                correct = matched == p.expected
                c_icon  = "[green]✓[/green]" if correct else "[red]✗[/red]"
                m_icon  = "[green]✅[/green]" if matched  else "[red]❌[/red]"
                dt.add_row(
                    p.id, p.label,
                    f"{score:.4f}", m_icon, c_icon,
                    f"{p.elapsed_s}s",
                    p.result.get("method", "?")
                )
            console.print(dt)
    else:
        print(f"\n{'='*55}")
        print(f"OMNI-RECO v2.0 — Régression : {rep.accuracy*100:.1f}% accuracy")
        print(f"TP={rep.same_tp} FN={rep.same_fn} TN={rep.diff_tn} FP={rep.diff_fp}")
        if rep.calibrated_threshold:
            print(f"Seuil calibré (Youden) : {rep.calibrated_threshold}")
        print(f"{'='*55}\n")


def save_report(pairs: list, rep: RegressionReport, path: str = "regression_report.json"):
    data = {
        "summary": asdict(rep),
        "pairs": [
            {
                "id": p.id, "label": p.label, "expected": p.expected,
                "desc": p.desc,
                "result": p.result,
                "error": p.error,
                "elapsed_s": p.elapsed_s,
            }
            for p in pairs
        ]
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

async def _main(args):
    pairs = TEST_PAIRS.copy()

    if args.dataset:
        # Chargement d'un dataset externe JSON
        with open(args.dataset) as f:
            raw = json.load(f)
        pairs = [TestPair(**p) for p in raw["pairs"]]

    if HAS_RICH:
        console.print(f"[cyan]Lancement de {len(pairs)} test(s)...[/cyan]")

    results = await run_all(pairs, model_name=args.model)
    rep     = analyze(results)
    print_report(results, rep, detailed=args.detailed)

    out = save_report(results, rep, "regression_report.json")
    if HAS_RICH:
        console.print(f"\n[dim]Rapport JSON sauvegardé : {out}[/dim]")
    else:
        print(f"Rapport : {out}")


def main():
    parser = argparse.ArgumentParser(description="OMNI-RECO v2.0 — Tests de régression")
    parser.add_argument("--detailed",   action="store_true", help="Affiche le détail par paire")
    parser.add_argument("--dataset",    default=None,        help="Fichier JSON dataset externe")
    parser.add_argument("--model",      default="buffalo_l", help="Modèle InsightFace")
    args = parser.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()

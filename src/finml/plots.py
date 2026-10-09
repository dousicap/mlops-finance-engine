"""Graphiques du tournoi et du champion, enregistrés dans MLflow sous « graphiques/ ».

- classement des candidats : moyenne de validation croisée et écart-type, champion en couleur ;
- classification : matrice de confusion du champion au seuil servi, courbes ROC et
  précision-rappel (binaire) ;
- régression : valeurs prédites contre valeurs réelles, distribution des résidus.

Tout est calculé sur le jeu de test, jamais vu pendant la sélection.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # rendu sans écran (CI, conteneur)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

ACCENT, NEUTRAL = "#2a6f97", "#b8c4ce"


def leaderboard(board: list[dict], metric: str, champion: str | None = None):
    champion = champion or board[0]["candidat"]
    names = [r["candidat"] for r in board][::-1]
    means = np.array([r["cv_moyenne"] for r in board])[::-1]
    stds = np.array([r["cv_ecart_type"] for r in board])[::-1]
    colors = [ACCENT if n == champion else NEUTRAL for n in names]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(board) + 1.2))
    ax.barh(names, means, xerr=stds, color=colors, capsize=3)
    lo = max(0.0, float((means - stds).min()) - 0.05) if metric not in {"rmse", "mae"} else 0.0
    ax.set_xlim(left=lo)
    ax.set_xlabel(f"{metric}, moyenne de validation croisée (barre = écart-type)")
    ax.set_title("Tournoi des modèles : le champion en bleu")
    for y, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s, y, f" {m:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    return fig


def confusion(y_true, y_pred, labels):
    fig, ax = plt.subplots(figsize=(min(4 + 0.4 * len(labels), 12),) * 2)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, ax=ax, cmap="Blues",
                                            colorbar=False, xticks_rotation=45)
    ax.set_title("Matrice de confusion du champion (jeu de test)")
    ax.set_xlabel("classe prédite")
    ax.set_ylabel("classe réelle")
    fig.tight_layout()
    return fig


def binary_curves(y_true, proba):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.5))
    RocCurveDisplay.from_predictions(y_true, proba, ax=a, color=ACCENT)
    a.plot([0, 1], [0, 1], ls="--", color=NEUTRAL)
    a.set_title("Courbe ROC")
    PrecisionRecallDisplay.from_predictions(y_true, proba, ax=b, color=ACCENT)
    b.axhline(float(np.mean(y_true)), ls="--", color=NEUTRAL)
    b.set_title("Courbe précision-rappel")
    fig.tight_layout()
    return fig


def calibration(y_true, proba):
    """Probabilité prédite contre fréquence observée : la diagonale = probabilités fiables."""
    fig, (a, b) = plt.subplots(2, 1, figsize=(6, 6.5), height_ratios=[3, 1], sharex=True)
    CalibrationDisplay.from_predictions(y_true, proba, n_bins=10, strategy="quantile", ax=a, color=ACCENT, name="champion")
    a.legend(a.get_lines(), ["calibration parfaite", "champion"], loc="lower right")
    a.set_title("Calibration des probabilités (diagonale = fiable)")
    a.set_xlabel("")
    a.set_ylabel("fréquence réelle des positifs")
    b.hist(proba, bins=30, color=NEUTRAL)
    b.set_xlabel("probabilité prédite")
    b.set_ylabel("effectif")
    fig.tight_layout()
    return fig


def threshold_metrics(y_true, proba, served: float):
    """Précision, rappel et spécificité selon le seuil : l'arbitrage faux positifs / faux négatifs."""
    y_true, proba = np.asarray(y_true).astype(int), np.asarray(proba)
    grid = np.unique(np.r_[np.linspace(0.001, 0.999, 200), min(served, 1.0)])  # le seuil servi y figure
    prec, rec, spec = [], [], []
    for t in grid:
        flag = proba >= t
        tp = np.sum(flag & (y_true == 1))
        prec.append(tp / max(flag.sum(), 1))
        rec.append(tp / max(np.sum(y_true == 1), 1))
        spec.append(np.sum(~flag & (y_true == 0)) / max(np.sum(y_true == 0), 1))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(grid, prec, label="précision", color=ACCENT)
    ax.plot(grid, rec, label="rappel (sensibilité)", color="#c8553d")
    ax.plot(grid, spec, label="spécificité", color="#588157")
    ax.axvline(served, ls="--", color="#333333")
    ax.text(served, 1.02, f" seuil métier {served:.3g}", fontsize=8)
    ax.set_xlabel("seuil de décision")
    ax.set_ylim(0, 1.08)
    ax.set_title("Métriques selon le seuil")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def regression(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.5))
    a.scatter(y_true, y_pred, s=10, alpha=0.6, color=ACCENT)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    a.plot(lims, lims, ls="--", color=NEUTRAL)
    a.set_xlabel("valeur réelle")
    a.set_ylabel("valeur prédite")
    a.set_title("Prédit contre réel (diagonale = parfait)")
    b.hist(y_true - y_pred, bins=30, color=ACCENT)
    b.set_xlabel("résidu (réel - prédit)")
    b.set_title("Distribution des résidus")
    fig.tight_layout()
    return fig

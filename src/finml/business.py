"""Traduction des prédictions en décisions et en euros.

Deux logiques métier, déclarées dans le bloc `business` de la configuration :

- cost : classification binaire (ex. facture payée en retard). Le seuil de décision n'est pas
  0,5 par défaut : il minimise le coût attendu, compte tenu du coût d'un retard non anticipé
  et du coût d'une relance préventive inutile.
- automation : classification multi-classes (ex. imputation comptable). Seules les propositions
  dont la confiance dépasse un seuil sont passées automatiquement. Le seuil est le plus bas qui
  garantit la précision cible. Le reste part en revue humaine.

Les seuils sont choisis sur les prédictions hors échantillon de la validation croisée, puis
mesurés sur le jeu de test, jamais vu pendant le choix.
"""

from __future__ import annotations

import numpy as np

from finml.config import AutomationBusiness, CostBusiness

THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def _cost(y: np.ndarray, flag: np.ndarray, b: CostBusiness) -> float:
    fn = np.sum((y == 1) & ~flag)
    fp = np.sum((y == 0) & flag)
    return float(fn * b.cost_false_negative + fp * b.cost_false_positive)


def best_cost_threshold(y, proba, b: CostBusiness) -> float:
    y, proba = np.asarray(y), np.asarray(proba)
    costs = [_cost(y, proba >= t, b) for t in THRESHOLD_GRID]
    return float(THRESHOLD_GRID[int(np.argmin(costs))])


def cost_report(y, proba, threshold: float, b: CostBusiness) -> dict[str, float]:
    """Coût par cas avec le modèle, comparé aux deux politiques sans modèle."""
    y, proba = np.asarray(y), np.asarray(proba)
    n = len(y)
    flag = proba >= threshold
    model = _cost(y, flag, b) / n
    nothing = _cost(y, np.zeros(n, bool), b) / n       # aucune relance préventive
    everyone = _cost(y, np.ones(n, bool), b) / n       # relance préventive de toutes les factures
    baseline = min(nothing, everyone)
    tp = np.sum((y == 1) & flag)
    return {
        "metier_seuil": threshold,
        "metier_part_signalee": float(flag.mean()),
        "metier_rappel": float(tp / max(np.sum(y == 1), 1)),
        "metier_precision": float(tp / max(flag.sum(), 1)),
        "metier_cout_par_cas_modele": model,
        "metier_cout_par_cas_sans_modele": baseline,
        "metier_economie_annuelle": (baseline - model) * b.annual_volume,
    }


def best_automation_threshold(y, pred, confidence, b: AutomationBusiness) -> float:
    """Seuil de confiance le plus bas dont la précision, sur les cas acceptés, atteint la cible."""
    y, pred, conf = np.asarray(y), np.asarray(pred), np.asarray(confidence)
    order = np.argsort(-conf)
    correct = (y[order] == pred[order]).astype(float)
    precision_at_k = np.cumsum(correct) / np.arange(1, len(correct) + 1)
    ok = np.where(precision_at_k >= b.target_precision)[0]
    if len(ok) == 0:
        return 1.01  # aucun seuil n'atteint la cible : tout part en revue humaine
    return float(conf[order][ok.max()])


def automation_report(y, pred, confidence, threshold: float, b: AutomationBusiness) -> dict[str, float]:
    y, pred, conf = np.asarray(y), np.asarray(pred), np.asarray(confidence)
    auto = conf >= threshold
    coverage = float(auto.mean())
    precision = float((y[auto] == pred[auto]).mean()) if auto.any() else 0.0
    return {
        "metier_seuil": threshold,
        "metier_taux_automatisation": coverage,
        "metier_precision_automatique": precision,
        "metier_heures_economisees_par_an": coverage * b.annual_volume * b.minutes_per_item / 60,
    }

"""Règles métier : seuil au coût minimal, seuil d'automatisation, tournoi."""

import numpy as np
import pytest

from finml.business import (
    automation_report,
    best_automation_threshold,
    best_cost_threshold,
    cost_report,
)
from finml.config import AutomationBusiness, CostBusiness, load_config

COST = CostBusiness(kind="cost", cost_false_negative=40, cost_false_positive=15, annual_volume=8000)


def test_cost_threshold_separates_perfect_scores():
    y = np.array([0] * 50 + [1] * 50)
    proba = np.array([0.1] * 50 + [0.9] * 50)
    t = best_cost_threshold(y, proba, COST)
    assert 0.1 < t <= 0.9
    report = cost_report(y, proba, t, COST)
    assert report["metier_cout_par_cas_modele"] == 0
    assert report["metier_economie_annuelle"] > 0


def test_expensive_misses_push_threshold_down():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    proba = np.clip(0.5 * y + rng.normal(0.25, 0.2, 2000), 0, 1)
    cheap = best_cost_threshold(y, proba, COST)
    costly = best_cost_threshold(y, proba, COST.model_copy(update={"cost_false_negative": 400}))
    assert costly < cheap


def test_automation_threshold_meets_target_precision():
    rng = np.random.default_rng(1)
    conf = rng.random(1000)
    y = np.array(["a"] * 1000)
    pred = np.where(rng.random(1000) < conf, "a", "b")  # plus la confiance est haute, plus c'est juste
    b = AutomationBusiness(kind="automation", target_precision=0.9, annual_volume=1000, minutes_per_item=2)
    t = best_automation_threshold(y, pred, conf, b)
    report = automation_report(y, pred, conf, t, b)
    assert report["metier_precision_automatique"] >= 0.9
    assert 0 < report["metier_taux_automatisation"] < 1


def test_unreachable_target_sends_everything_to_review():
    b = AutomationBusiness(kind="automation", target_precision=1.0, annual_volume=10, minutes_per_item=1)
    t = best_automation_threshold(["a", "b"], ["b", "a"], [0.9, 0.8], b)
    assert automation_report(["a", "b"], ["b", "a"], [0.9, 0.8], t, b)["metier_taux_automatisation"] == 0


def test_cost_block_rejected_on_text_usecase(project):
    raw = load_config("imputation_comptable").model_dump(by_alias=True)
    raw["business"] = COST.model_dump()
    with pytest.raises(ValueError):
        type(load_config("imputation_comptable")).model_validate(raw)


def test_tournament_ranks_by_cross_validation_and_skips_missing_deps(project, monkeypatch):
    from finml import data, train

    monkeypatch.setattr(train, "_skip_reason", lambda cand, n: "absent" if cand.type == "torch_mlp" else None)
    data.generate(load_config("retard_paiement"))
    result = train.train("retard_paiement")
    cv = [row["cv_moyenne"] for row in result.leaderboard]
    assert cv == sorted(cv, reverse=True)
    assert result.best_candidate == result.leaderboard[0]["candidat"]
    assert "reseau_neurones_pytorch" not in {row["candidat"] for row in result.leaderboard}
    # tous les candidats sauf PyTorch, plus la référence naïve
    assert len(result.leaderboard) == len(load_config("retard_paiement").candidates)
    assert "reference_naive" in {row["candidat"] for row in result.leaderboard}
    assert "metier_economie_annuelle" in result.business_metrics


def test_threshold_is_exact_not_on_a_grid():
    """Coût très asymétrique : le seuil optimal peut être sous 0,05, une grille le manquerait."""
    y = np.array([0] * 900 + [1] * 100)
    proba = np.r_[np.full(900, 0.001), np.full(100, 0.02)]
    t = best_cost_threshold(y, proba, COST)
    assert t == 0.02
    assert cost_report(y, proba, t, COST)["metier_rappel"] == 1.0


def test_capacity_limits_the_share_flagged():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 1000)
    proba = np.clip(0.4 * y + rng.random(1000) * 0.6, 0, 1)
    capped = COST.model_copy(update={"cost_false_negative": 400, "capacity": 0.2})
    t = best_cost_threshold(y, proba, capped)
    report = cost_report(y, proba, t, capped)
    assert report["metier_part_signalee"] <= 0.2
    assert report["metier_cout_par_cas_sans_modele"] == 400 * y.mean()  # seule politique réalisable


def test_no_action_when_model_cannot_beat_doing_nothing():
    y = np.array([0, 0, 0, 1])
    proba = np.array([0.9, 0.8, 0.7, 0.1])  # le modèle classe à l'envers
    assert best_cost_threshold(y, proba, COST) > 1

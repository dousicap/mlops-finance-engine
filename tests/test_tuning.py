"""Recherche d'hyperparamètres (Optuna) et règle de simplicité."""

import pytest
import yaml
from pydantic import ValidationError

from finml.config import SearchParam, load_config
from finml.tuning import pick_champion


def test_search_param_needs_exactly_one_kind():
    assert SearchParam.model_validate({"int": [1, 5]}).int_range == (1, 5)
    with pytest.raises(ValidationError):
        SearchParam.model_validate({"int": [1, 5], "float": [0.1, 1.0]})
    with pytest.raises(ValidationError):
        SearchParam.model_validate({"log": True})


def _row(name, typ, cv, p):
    return {"candidat": name, "type": typ, "cv_moyenne": cv, "p_value_vs_champion": p}


def test_simpler_model_wins_a_statistical_tie(project):
    cfg = load_config("retard_paiement")
    board = [_row("xgb", "xgboost", 0.800, None), _row("rf", "random_forest", 0.795, 0.40),
             _row("lr", "logistic_regression", 0.794, 0.30), _row("knn", "knn", 0.70, 0.001)]
    champion, note = pick_champion(cfg, board)
    assert champion == "lr" and "égalité statistique" in note


def test_no_tie_when_gap_is_significant_or_too_large(project):
    cfg = load_config("retard_paiement")
    significant = [_row("xgb", "xgboost", 0.80, None), _row("lr", "logistic_regression", 0.799, 0.01)]
    too_large = [_row("xgb", "xgboost", 0.80, None), _row("lr", "logistic_regression", 0.78, 0.30)]
    assert pick_champion(cfg, significant) == ("xgb", None)
    assert pick_champion(cfg, too_large) == ("xgb", None)  # au-delà de la tolérance de 0,01


def test_tuning_never_degrades_and_traces_trials(project, monkeypatch):
    """Diabète, 4 essais par candidat : CV ≥ réglage de départ, essais tracés dans MLflow."""
    import mlflow

    from finml.train import train

    cfg_file = project / "usecases" / "diabete" / "config.yaml"
    cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    cfg["candidates"] = [c for c in cfg["candidates"] if c["type"] in {"ridge", "lasso", "random_forest"}]
    cfg["tuning"] = {"n_trials": 50, "timeout_seconds": 30}
    cfg_file.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    monkeypatch.setenv("FINML_TUNING_TRIALS", "4")  # le budget CI remplace celui de la config

    result = train("diabete")
    tuned = [r for r in result.leaderboard if r["type"] != "dummy"]
    assert all(r["essais"] == 4 for r in tuned)
    assert all(r["cv_moyenne"] >= r["cv_depart"] for r in tuned)
    exp = mlflow.get_experiment_by_name("finml-diabete")
    trials = mlflow.search_runs([exp.experiment_id], filter_string="tags.etat = 'complet'")
    assert len(trials) + len(mlflow.search_runs([exp.experiment_id], filter_string="tags.etat = 'elague'")) == 12


def test_untuned_wins_a_tie_at_equal_complexity(project):
    """Cas réel de la maintenance : XGBoost réglé et boosting par histogramme à égalité."""
    cfg = load_config("maintenance_predictive")
    board = [{**_row("xgboost", "xgboost", 0.8079, None), "reglage": "optuna"},
             {**_row("hgb", "hist_gradient_boosting", 0.8015, 0.63), "reglage": "depart"},
             {**_row("rf", "random_forest", 0.7915, 0.32), "reglage": "optuna"}]
    champion, note = pick_champion(cfg, board)
    assert champion == "hgb" and "réglage de départ" in note


def test_tuning_kept_only_when_clearly_better(project):
    import numpy as np

    from finml.train import _clearly_better

    cfg = load_config("retard_paiement")
    start = np.array([0.735, 0.738, 0.733, 0.737, 0.736])
    assert not _clearly_better(cfg, start + 0.0005, start)  # gain dans le bruit : égalité
    assert _clearly_better(cfg, start + 0.02, start)  # au-delà de la tolérance
    assert not _clearly_better(cfg, start - 0.02, start)  # moins bon

"""Moteur générique : détection par finml init, régression et multi-classes de bout en bout."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_wine

from finml import data
from finml.api import create_app
from finml.config import load_config
from finml.export import export_champion
from finml.scaffold import infer, init_usecase
from finml.train import train


def test_infer_detects_task_and_column_types():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "id": np.arange(n),
        "age": rng.integers(18, 80, n),
        "ville": rng.choice(["paris", "lyon", "dakar"], n),
        "commentaire": [f"client satisfait du service numero {i} et de la livraison" for i in range(n)],
        "constante": 1,
        "montant": rng.normal(100, 20, n).round(2),
        "segment": rng.choice(["a", "b", "c"], n),
        "achete": rng.integers(0, 2, n),
    })
    reg = infer(df, "montant")
    assert reg["task"] == "tabular_regression"
    assert reg["excluded"]["id"].startswith("identifiant") and reg["excluded"]["constante"] == "constante"
    assert "commentaire" in reg["excluded"]
    assert set(reg["categorical"]) == {"ville", "segment"}
    assert infer(df, "achete")["n_classes"] == 2
    assert infer(df, "segment")["task"] == "tabular_classification"
    text_only = infer(df[["commentaire", "segment"]], "segment")
    assert text_only["task"] == "text_classification" and text_only["text"] == ["commentaire"]


def test_init_refuses_to_overwrite(project, tmp_path):
    csv = tmp_path / "x.csv"
    pd.DataFrame({"a": range(50), "b": [0, 1] * 25, "y": np.linspace(0, 1, 50)}).to_csv(csv, index=False)
    init_usecase("essai", csv, "y")
    with pytest.raises(FileExistsError):
        init_usecase("essai", csv, "y")


def test_multiclass_tabular_end_to_end_from_csv(project, tmp_path):
    """Trois classes de vin, valeurs manquantes injectées : init, entraînement, API."""
    wine = load_wine(as_frame=True).frame.rename(columns={"target": "cepage"})
    wine.loc[::17, "alcohol"] = np.nan
    csv = tmp_path / "vins.csv"
    wine.to_csv(csv, index=False)
    _, info = init_usecase("vins", csv, "cepage")
    assert info["task"] == "tabular_classification" and info["n_classes"] == 3
    result = train("vins")
    assert result.metric == "accuracy" and result.score > 0.85
    ranked = {r["candidat"] for r in result.leaderboard}
    assert "reseau_neurones_pytorch" not in ranked  # limité au binaire : ignoré, pas bloquant
    export_champion("vins", project / "models")
    record = wine.drop(columns="cepage").iloc[0].to_dict()
    record["alcohol"] = None  # valeur manquante à l'appel : imputée par le pipeline
    with TestClient(create_app(models_dir=project / "models", usecases=["vins"])) as client:
        pred = client.post("/predict/vins", json={"records": [record]}).json()["predictions"][0]
    assert pred["classe"] == "0" and len(pred["top3"]) == 3 and 0 < pred["confiance"] <= 1


def test_regression_case_serves_values_and_logs_charts(trained):
    cfg = load_config("diabete")
    assert cfg.task == "tabular_regression"
    row = data.load(cfg).drop(columns=cfg.target).iloc[0].to_dict()
    with TestClient(create_app(models_dir=trained / "models", prediction_log_dir=trained / "logs")) as client:
        r = client.post("/predict/diabete", json={"records": [row]})
    assert r.status_code == 200 and isinstance(r.json()["predictions"][0]["valeur"], float)


@pytest.mark.parametrize("usecase,expected", [
    ("diabete", {"classement_tournoi.png", "predit_contre_reel.png"}),
    ("retard_paiement", {"classement_tournoi.png", "matrice_confusion.png", "courbes_roc_precision_rappel.png",
                         "calibration.png", "metriques_selon_seuil.png"}),
    ("imputation_comptable", {"classement_tournoi.png", "matrice_confusion.png"}),
])
def test_charts_are_logged(trained, usecase, expected):
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{trained / 'mlflow.db'}")
    exp = mlflow.get_experiment_by_name(f"finml-{usecase}")
    run = mlflow.search_runs([exp.experiment_id], filter_string="tags.mlflow.runName LIKE 'tournoi%'").iloc[0]
    names = {a.path.split("/")[-1] for a in mlflow.MlflowClient().list_artifacts(run.run_id, "graphiques")}
    assert expected <= names


def test_binary_text_target_is_encoded(project, tmp_path):
    """Cible « oui » / « non » : codée 0/1 avec « oui » positif, entraînement complet."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=600)
    df = pd.DataFrame({"x": x, "bruit": rng.normal(size=600),
                       "resilie": np.where(x + rng.normal(0, 0.5, 600) > 0, "oui", "non")})
    csv = tmp_path / "resiliation.csv"
    df.to_csv(csv, index=False)
    init_usecase("resiliation", csv, "resilie")
    result = train("resiliation")
    assert result.metric == "roc_auc" and result.score > 0.8
    export_champion("resiliation", project / "models")
    with TestClient(create_app(models_dir=project / "models", usecases=["resiliation"])) as client:
        preds = client.post("/predict/resiliation", json={"records": [{"x": 3.0, "bruit": 0.0},
                                                                      {"x": -3.0, "bruit": 0.0}]}).json()
    assert [p["classe"] for p in preds["predictions"]] == ["oui", "non"]


def test_naive_baseline_must_be_beaten(project, tmp_path):
    """Cible indépendante des variables : aucun modèle ne bat la référence naïve, rien n'est publié."""
    from finml.train import QualityGateError

    rng = np.random.default_rng(4)
    df = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400), "y": rng.normal(size=400)})
    csv = tmp_path / "bruit.csv"
    df.to_csv(csv, index=False)
    init_usecase("bruit", csv, "y")
    with pytest.raises(QualityGateError):
        train("bruit")


def test_leakage_is_flagged():
    from finml.config import UseCaseConfig
    from finml.evaluation import leakage_check

    rng = np.random.default_rng(5)
    y = pd.Series(rng.integers(0, 2, 500))
    X = pd.DataFrame({"honnete": rng.normal(size=500) + y * 0.5, "fuite": y * 10 + rng.normal(0, 0.1, 500)})
    cfg = UseCaseConfig.model_validate({
        "name": "t", "description": "t", "task": "tabular_classification", "registered_model": "t",
        "target": "y", "features": {"numeric": ["honnete", "fuite"]}, "data": {"path": "x.csv"},
        "metric": "roc_auc", "min_score": 0.5, "candidates": [{"name": "lr", "type": "logistic_regression"}]})
    assert list(leakage_check(cfg, X, y)) == ["fuite"]


def test_full_evaluation_is_consistent(trained):
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{trained / 'mlflow.db'}")
    exp = mlflow.get_experiment_by_name("finml-retard_paiement")
    run = mlflow.search_runs([exp.experiment_id], filter_string="tags.mlflow.runName LIKE 'tournoi%'").iloc[0]
    e = {k.replace("metrics.eval_", ""): v for k, v in run.items() if k.startswith("metrics.eval_")}
    n_test = e["vrais_positifs"] + e["faux_positifs"] + e["faux_negatifs"] + e["vrais_negatifs"]
    assert n_test == 600  # 20 % de 3 000 factures
    assert e["specificite"] == pytest.approx(e["vrais_negatifs"] / (e["vrais_negatifs"] + e["faux_positifs"]))
    assert e["rappel"] == pytest.approx(e["vrais_positifs"] / (e["vrais_positifs"] + e["faux_negatifs"]))
    assert -1 <= e["mcc"] <= 1 and e["gini"] == pytest.approx(2 * e["roc_auc"] - 1)
    assert e["roc_auc_ic95_bas"] < e["roc_auc"] < e["roc_auc_ic95_haut"]


def test_corrected_t_test():
    from finml.evaluation import corrected_t_test

    same = np.array([0.80, 0.81, 0.79, 0.80, 0.82])
    assert corrected_t_test(same, same, 0.25) == 1.0
    assert corrected_t_test(same + 0.10, same + np.array([0, 0.001, 0, -0.001, 0]), 0.25) < 0.01
    noisy = same + np.array([0.02, -0.03, 0.01, 0.02, -0.01])
    assert corrected_t_test(noisy, same, 0.25) > 0.05  # écart dans le bruit : pas de différence


def test_leakage_by_combination_is_flagged(project):
    """Piège réel : les modes de panne d'AI4I, chacun partiel, reconstituent la panne ensemble."""
    from finml.evaluation import leakage_check

    raw = pd.read_csv(project / "usecases/maintenance_predictive/source/ai4i2020.csv.gz")
    cols = ["Torque [Nm]", "Tool wear [min]", "TWF", "HDF", "PWF", "OSF", "RNF"]
    cfg = load_config("maintenance_predictive")
    cfg = cfg.model_copy(update={"features": cfg.features.model_copy(update={"numeric": cols, "categorical": []})})
    flagged = leakage_check(cfg, raw[cols], raw["Machine failure"])
    assert len(flagged) == 1 and next(iter(flagged)).startswith("combinaison")
    prepared = load_config("maintenance_predictive")
    data.generate(prepared)
    df = data.load(prepared)
    assert leakage_check(prepared, df[prepared.features.all], df[prepared.target]) == {}

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
    ("retard_paiement", {"classement_tournoi.png", "matrice_confusion.png", "courbes_roc_precision_rappel.png"}),
    ("imputation_comptable", {"classement_tournoi.png", "matrice_confusion.png"}),
])
def test_charts_are_logged(trained, usecase, expected):
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{trained / 'mlflow.db'}")
    exp = mlflow.get_experiment_by_name(f"finml-{usecase}")
    run = mlflow.search_runs([exp.experiment_id], filter_string="tags.mlflow.runName LIKE 'tournoi%'").iloc[0]
    names = {a.path.split("/")[-1] for a in mlflow.MlflowClient().list_artifacts(run.run_id, "graphiques")}
    assert expected <= names

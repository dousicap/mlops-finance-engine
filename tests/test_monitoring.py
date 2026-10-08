"""Sérialisation sûre, métriques, journal des prédictions, dérive, prédiction par lots."""

import json

import pandas as pd
import pytest
import skops.io as sio
from fastapi.testclient import TestClient

from finml.api import create_app
from finml.cli import main as cli
from finml.config import load_config
from finml.models import UntrustedModelError, load_estimator
from finml.monitoring import detect_drift, load_production_log, simulate_records


class Piege:
    """Classe absente de la liste blanche : skops doit refuser de la recharger."""

    def __init__(self):
        self.x = 1


def test_untrusted_type_is_refused(tmp_path):
    path = tmp_path / "piege.skops"
    sio.dump(Piege(), path)
    with pytest.raises(UntrustedModelError):
        load_estimator(path)


def test_exported_models_contain_no_pickle(trained):
    for name in ("retard_paiement", "imputation_comptable"):
        model_dir = trained / "models" / name
        assert (model_dir / "artifacts" / "estimator.skops").exists()
        assert not list(model_dir.rglob("*.pkl"))
        assert "code_model" in (model_dir / "MLmodel").read_text()


def _reference(project, name):
    return pd.read_csv(project / "models" / name / "artifacts" / "reference.csv")


@pytest.mark.parametrize("name", ["retard_paiement", "imputation_comptable"])
def test_drift_detection(trained, name):
    cfg = load_config(name)
    reference = _reference(trained, name)
    with TestClient(create_app(models_dir=trained / "models", prediction_log_dir=trained / "logs")) as client:
        for drift, seed in ((False, 1), (True, 2)):
            records = simulate_records(cfg, 300, drift=drift, seed=seed)
            for i in range(0, len(records), 100):
                assert client.post(f"/predict/{name}", json={"records": records[i:i + 100]}).status_code == 200
            log = load_production_log(trained / "logs" / f"{name}.jsonl")
            report = detect_drift(cfg, reference, log.tail(300))
            assert report["derive"] is drift, report["variables_en_derive"]


def test_insufficient_production_data(trained):
    cfg = load_config("retard_paiement")
    production = _reference(trained, "retard_paiement").head(20)
    assert detect_drift(cfg, _reference(trained, "retard_paiement"), production)["statut"] == "donnees_insuffisantes"


def test_metrics_and_prediction_log(trained, capsys):
    cfg = load_config("retard_paiement")
    records = simulate_records(cfg, 7, drift=False)
    with TestClient(create_app(models_dir=trained / "models", prediction_log_dir=trained / "logs")) as client:
        assert client.post("/predict/retard_paiement", json={"records": records}).status_code == 200
        client.post("/predict/retard_paiement", json={"records": [{"montant_ttc": 1}]})
        metrics = client.get("/metrics/").text
    assert 'finml_predictions_total{usecase="retard_paiement"}' in metrics
    assert 'finml_prediction_errors_total{raison="donnees_invalides",usecase="retard_paiement"}' in metrics
    assert "finml_prediction_latency_seconds_bucket" in metrics
    stdout_lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("{")]
    assert len(stdout_lines) == 7 and json.loads(stdout_lines[0])["usecase"] == "retard_paiement"
    lines = (trained / "logs" / "retard_paiement.jsonl").read_text().splitlines()
    assert len(lines) == 7
    entry = json.loads(lines[0])
    assert entry["model_version"] == 1 and "probabilite" in entry["outputs"]


def test_batch_prediction(trained):
    cfg = load_config("imputation_comptable")
    pd.DataFrame({"libelle": ["Cabinet Durand Associes note honoraires", "PRLV SEPA FibreNet box fibre"]}).to_csv(
        trained / "lot.csv", index=False)
    cli(["batch", cfg.name, str(trained / "lot.csv"), str(trained / "resultat.csv")])
    out = pd.read_csv(trained / "resultat.csv", dtype={"compte": str})
    assert out["compte"].tolist() == ["6226", "626"]
    assert {"confiance", "a_revoir", "modele", "predit_le"} <= set(out.columns)

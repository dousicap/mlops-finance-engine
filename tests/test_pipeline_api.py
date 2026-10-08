"""Bout en bout : génération -> entraînement -> Registry -> export -> API."""

import json

import pytest
from conftest import set_config
from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient

from finml import data
from finml.api import create_app
from finml.config import load_config
from finml.train import QualityGateError, train

FACTURE = {
    "montant_ttc": 12500.0, "delai_contractuel_jours": 60, "mois_emission": 8,
    "anciennete_client_mois": 4, "nb_retards_12m": 6, "retard_moyen_12m_jours": 35,
    "relance_automatique": 0, "secteur": "btp", "taille_client": "tpe", "mode_paiement": "cheque",
}
FACTURE_SURE = {
    **FACTURE, "montant_ttc": 300.0, "delai_contractuel_jours": 30, "mois_emission": 3,
    "anciennete_client_mois": 96, "nb_retards_12m": 0, "retard_moyen_12m_jours": 0,
    "relance_automatique": 1, "secteur": "sante", "taille_client": "eti", "mode_paiement": "prelevement",
}


def test_training_registers_champion(trained):
    client = MlflowClient()
    for name in ("retard_paiement", "imputation_comptable"):
        cfg = load_config(name)
        mv = client.get_model_version_by_alias(cfg.registered_model, "champion")
        assert int(mv.version) == 1
        manifest = json.loads((trained / "models" / name / "manifest.json").read_text())
        assert manifest["score"] >= cfg.min_score
        assert len(manifest["data_sha256"]) == 64


def test_retrain_without_gain_becomes_challenger(trained):
    result = train("retard_paiement")  # mêmes données, même score : pas de promotion
    assert result.alias == "challenger"
    cfg = load_config("retard_paiement")
    champion = MlflowClient().get_model_version_by_alias(cfg.registered_model, "champion")
    assert int(champion.version) == 1


def test_quality_gate_blocks_weak_model(project):
    set_config(project, "retard_paiement", min_score=0.99)
    data.generate(load_config("retard_paiement"))
    with pytest.raises(QualityGateError):
        train("retard_paiement")


def test_api_end_to_end(trained):
    with TestClient(create_app(models_dir=trained / "models")) as client:
        assert client.get("/health").status_code == 200
        names = [u["nom"] for u in client.get("/usecases").json()]
        assert names == ["imputation_comptable", "retard_paiement"]

        r = client.post("/predict/retard_paiement", json={"records": [FACTURE, FACTURE_SURE]})
        assert r.status_code == 200
        risky, safe = r.json()["predictions"]
        assert risky["probabilite"] > safe["probabilite"]
        assert risky["prediction"] == 1

        r = client.post("/predict/imputation_comptable",
                        json={"records": [{"libelle": "PRLV SEPA FibreNet abonnement internet"},
                                          {"libelle": "Cabinet Durand Associes note honoraires"}]})
        assert r.status_code == 200
        preds = r.json()["predictions"]
        assert [p["compte"] for p in preds] == ["626", "6226"]
        assert preds[1]["intitule"] == "Honoraires"
        assert len(preds[0]["top3"]) == 3


def test_api_rejects_bad_input(trained):
    with TestClient(create_app(models_dir=trained / "models")) as client:
        bad = {**FACTURE, "mois_emission": 14}
        assert client.post("/predict/retard_paiement", json={"records": [bad]}).status_code == 422
        missing = {k: v for k, v in FACTURE.items() if k != "secteur"}
        assert client.post("/predict/retard_paiement", json={"records": [missing]}).status_code == 422
        texte = {**FACTURE, "montant_ttc": "douze mille"}
        r = client.post("/predict/retard_paiement", json={"records": [texte]})
        assert r.status_code == 422 and "non numérique" in str(r.json())
        assert client.post("/predict/inconnu", json={"records": [FACTURE]}).status_code == 404


def test_health_is_503_without_models(project):
    with TestClient(create_app(models_dir=project / "vide")) as client:
        r = client.get("/health")
        assert r.status_code == 503
        assert r.json()["status"] == "degraded"

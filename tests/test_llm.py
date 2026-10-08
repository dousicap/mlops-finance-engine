"""Second avis LLM : garde-fous, panne du fournisseur, traces, endpoint."""

import json

import mlflow
import pytest
from fastapi.testclient import TestClient

from finml.api import create_app
from finml.config import load_config
from finml.llm import (
    EchoClient,
    LLMNotConfigured,
    LLMResponse,
    make_client,
    render_prompt,
    second_avis,
)

TOP3 = [{"compte": "626", "intitule": "Frais postaux et de télécommunications", "score": 0.41},
        {"compte": "6135", "intitule": "Locations mobilières", "score": 0.35},
        {"compte": "627", "intitule": "Services bancaires et assimilés", "score": 0.1}]
EXEMPLES = [{"libelle": "ABONNEMENT INTERNET FIBRENET", "compte": "626", "similarite": 0.8}]


class Broken:
    name = "panne"

    def complete(self, prompt):
        raise TimeoutError("fournisseur indisponible")


@pytest.fixture
def cfg(project):
    return load_config("imputation_comptable")


def test_prompt_lists_allowed_accounts_and_context(cfg):
    prompt = render_prompt(cfg, "PRLV TeleCom One 4521", TOP3, EXEMPLES)
    assert "- 6226 : Honoraires" in prompt
    assert "626 (Frais postaux" in prompt and "ABONNEMENT INTERNET FIBRENET" in prompt
    assert '{"compte": "<code>"' in prompt  # accolades du gabarit JSON préservées


@pytest.mark.parametrize("reply,reason", [
    ('{"compte": "999", "confiance": "haute", "justification": "x"}', "compte_hors_liste"),
    ("Je pense que c'est le 626.", "sortie_non_conforme"),
    ('{"compte": "626", "confiance": "certaine", "justification": "x"}', "sortie_non_conforme"),
])
def test_guardrails_reject_bad_outputs(cfg, reply, reason):
    result = second_avis(cfg, EchoClient(reply), "PRLV TeleCom One", TOP3, EXEMPLES)
    assert not result.valide and result.rejet == reason and result.avis is None


def test_provider_failure_does_not_raise(cfg):
    result = second_avis(cfg, Broken(), "PRLV TeleCom One", TOP3, EXEMPLES)
    assert not result.valide and result.rejet.startswith("appel_en_echec")


def test_valid_answer_is_traced_with_cost(cfg, monkeypatch):
    monkeypatch.setattr(cfg.llm, "price_input_per_mtok", 1.0)
    reply = '{"compte": "6135", "confiance": "moyenne", "justification": "Abonnement de location."}'
    client = EchoClient(reply)
    client.complete = lambda p: LLMResponse(reply, 1000, 50)
    from finml.train import default_tracking_uri

    mlflow.set_tracking_uri(default_tracking_uri())
    mlflow.set_experiment("test-llm")
    result = second_avis(cfg, client, "PRLV TeleCom One", TOP3, EXEMPLES)
    if hasattr(mlflow, "flush_trace_async_logging"):
        mlflow.flush_trace_async_logging()
    assert result.valide and result.avis.compte == "6135" and result.cout_eur == 0.001
    exp = mlflow.get_experiment_by_name("test-llm")
    traces = mlflow.search_traces(locations=[exp.experiment_id])
    assert len(traces) >= 1


def test_client_requires_key(cfg, monkeypatch):
    monkeypatch.delenv("FINML_LLM_API_KEY", raising=False)
    with pytest.raises(LLMNotConfigured):
        make_client(cfg)


def test_endpoint(trained, monkeypatch):
    monkeypatch.delenv("FINML_LLM_API_KEY", raising=False)
    with TestClient(create_app(models_dir=trained / "models", prediction_log_dir=trained / "logs")) as client:
        r = client.post("/second-avis/imputation_comptable", json={"libelle": "CB 48213"})
        assert r.status_code == 503  # aucune clé : désactivé proprement
        monkeypatch.setenv("FINML_LLM_PROVIDER", "echo")
        r = client.post("/second-avis/imputation_comptable", json={"libelle": "CB 48213"})
        body = r.json()
        assert r.status_code == 200 and body["modele"]["a_revoir"] is True
        assert body["second_avis"]["valide"] is True
        assert body["second_avis"]["compte"] == body["modele"]["top3"][0]["compte"]
        r = client.post("/second-avis/retard_paiement", json={"libelle": "x"})
        assert r.status_code == 400
        assert 'finml_llm_second_avis_total{resultat="valide",usecase="imputation_comptable"}' in \
            client.get("/metrics/").text
    assert json.dumps(body)  # réponse sérialisable

"""Second avis par LLM, tracé et évalué (LLMOps).

Déroulé pour une pièce envoyée en revue humaine :
1. prompt versionné rempli avec le libellé, les comptes autorisés, le top 3 du modèle et les
   écritures passées les plus proches ;
2. appel d'une API compatible OpenAI (Mistral par défaut), température 0, sortie JSON ;
3. garde-fous : JSON valide, compte dans la liste autorisée, sinon la proposition est rejetée ;
4. trace MLflow : prompt et version, modèle, entrées, sortie, tokens, latence, coût.

Le LLM ne décide jamais seul : sa proposition accompagne la pièce chez le comptable.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Literal, Protocol

import mlflow
from pydantic import BaseModel, ValidationError

from finml.config import UseCaseConfig, usecase_dir


class LLMNotConfigured(RuntimeError):
    """Aucun bloc llm dans la configuration, ou clé d'API absente."""


class SecondAvis(BaseModel):
    compte: str
    confiance: Literal["haute", "moyenne", "basse"]
    justification: str


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class SecondAvisResult:
    avis: SecondAvis | None
    valide: bool
    rejet: str | None
    latence_ms: float
    cout_eur: float
    prompt_version: str
    modele: str
    brut: str = field(repr=False, default="")

    def to_dict(self) -> dict:
        return {
            "compte": self.avis.compte if self.avis else None,
            "confiance": self.avis.confiance if self.avis else None,
            "justification": self.avis.justification if self.avis else None,
            "valide": self.valide,
            "rejet": self.rejet,
            "latence_ms": self.latence_ms,
            "cout_eur": self.cout_eur,
            "prompt_version": self.prompt_version,
            "modele": self.modele,
        }


class LLMClient(Protocol):
    name: str

    def complete(self, prompt: str) -> LLMResponse: ...


class OpenAICompatibleClient:
    """Client HTTP minimal pour /chat/completions (Mistral, OpenAI, vLLM, Ollama...)."""

    def __init__(self, base_url: str, model: str, api_key: str, temperature: float, timeout: float):
        self.base_url, self.name, self.api_key = base_url.rstrip("/"), model, api_key
        self.temperature, self.timeout = temperature, timeout

    def complete(self, prompt: str) -> LLMResponse:
        import httpx

        r = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.name,
                "temperature": self.temperature,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        body = r.json()
        usage = body.get("usage", {})
        return LLMResponse(
            body["choices"][0]["message"]["content"],
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )


class EchoClient:
    """Client hors ligne pour les tests et la CI : reprend la première proposition du modèle."""

    name = "echo-hors-ligne"

    def __init__(self, reply: str | None = None):
        self.reply = reply

    def complete(self, prompt: str) -> LLMResponse:
        if self.reply is not None:
            return LLMResponse(self.reply, len(prompt) // 4, 20)
        first = prompt.split("Propositions du modèle automatique, avec leur score :\n", 1)[1].split()[1]
        text = json.dumps({"compte": first, "confiance": "moyenne",
                           "justification": "Reprise de la première proposition du modèle."})
        return LLMResponse(text, len(prompt) // 4, 20)


def make_client(cfg: UseCaseConfig, provider: str = "auto") -> LLMClient:
    if cfg.llm is None:
        raise LLMNotConfigured(f"{cfg.name} : aucun bloc llm dans la configuration")
    if provider == "echo":
        return EchoClient()
    key = os.environ.get(cfg.llm.api_key_env)
    if not key:
        raise LLMNotConfigured(f"Variable {cfg.llm.api_key_env} absente : second avis désactivé")
    return OpenAICompatibleClient(cfg.llm.base_url, cfg.llm.model, key, cfg.llm.temperature,
                                  cfg.llm.timeout_seconds)


def render_prompt(cfg: UseCaseConfig, libelle: str, top3: list[dict], exemples: list[dict]) -> str:
    template = (usecase_dir(cfg.name) / cfg.llm.prompt).read_text(encoding="utf-8")
    comptes = "\n".join(f"- {code} : {label}" for code, label in cfg.labels.items())
    top = "\n".join(f"- {p['compte']} ({p.get('intitule', '')}) : {p['score']:.2f}" for p in top3) or "- aucune"
    ex = "\n".join(f"- « {e['libelle']} » -> {e['compte']}" for e in exemples) or "- aucune"
    return template.format(libelle=libelle, comptes=comptes, top3=top, exemples=ex)


def _parse(cfg: UseCaseConfig, text: str) -> tuple[SecondAvis | None, str | None]:
    try:
        avis = SecondAvis.model_validate_json(text)
    except ValidationError:
        return None, "sortie_non_conforme"
    if avis.compte not in cfg.labels:
        return None, "compte_hors_liste"
    return avis, None


def second_avis(cfg: UseCaseConfig, client: LLMClient, libelle: str, top3: list[dict],
                exemples: list[dict]) -> SecondAvisResult:
    prompt = render_prompt(cfg, libelle, top3, exemples)
    with mlflow.start_span(name="second_avis", span_type="LLM") as span:
        span.set_inputs({"libelle": libelle, "top3": top3})
        span.set_attributes({"prompt_version": cfg.llm.prompt_version, "modele": client.name,
                             "usecase": cfg.name})
        t0 = time.perf_counter()
        try:
            resp = client.complete(prompt)
        except Exception as e:  # noqa: BLE001 - un fournisseur indisponible ne bloque pas la revue
            resp, error = LLMResponse(""), f"appel_en_echec: {type(e).__name__}"
        else:
            error = None
        latency = round((time.perf_counter() - t0) * 1000, 1)
        avis, rejet = _parse(cfg, resp.text) if error is None else (None, error)
        cost = (resp.input_tokens * cfg.llm.price_input_per_mtok
                + resp.output_tokens * cfg.llm.price_output_per_mtok) / 1e6
        result = SecondAvisResult(avis, avis is not None, rejet, latency, round(cost, 6),
                                  cfg.llm.prompt_version, client.name, resp.text)
        span.set_attributes({"tokens_entree": resp.input_tokens, "tokens_sortie": resp.output_tokens,
                             "latence_ms": latency, "cout_eur": result.cout_eur, "valide": result.valide})
        span.set_outputs(result.to_dict())
    return result


def evaluate(cfg: UseCaseConfig, client: LLMClient, cases: list[dict]) -> dict[str, float]:
    """Évaluation hors ligne sur des cas étiquetés : {libelle, compte, top3, exemples}."""
    results = [second_avis(cfg, client, c["libelle"], c["top3"], c["exemples"]) for c in cases]
    n = len(results)
    valid = [r for r, c in zip(results, cases) if r.valide]
    correct = sum(1 for r, c in zip(results, cases) if r.valide and r.avis.compte == c["compte"])
    model_correct = sum(1 for c in cases if c["top3"] and c["top3"][0]["compte"] == c["compte"])
    return {
        "llm_cas": n,
        "llm_taux_sortie_valide": len(valid) / n if n else 0.0,
        "llm_exactitude": correct / n if n else 0.0,
        "modele_seul_exactitude": model_correct / n if n else 0.0,
        "llm_latence_moyenne_ms": sum(r.latence_ms for r in results) / n if n else 0.0,
        "llm_cout_total_eur": sum(r.cout_eur for r in results),
    }

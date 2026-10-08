"""API de prédiction unique pour tous les cas d'usage.

Source des modèles, par ordre de priorité :
1. FINML_MODELS_DIR/<cas>/ : modèle exporté et embarqué dans l'image (production) ;
2. MLflow Registry `models:/<nom>@champion` (développement, ou serveur MLflow partagé).

/health renvoie 503 tant qu'un modèle attendu n'est pas chargé : l'ALB ne route pas
de trafic vers un conteneur incapable de prédire.
/metrics expose les métriques Prometheus. Chaque prédiction est journalisée pour la
détection de dérive (voir finml.monitoring).
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic import BaseModel, Field

from finml.config import UseCaseConfig, list_usecases, load_config
from finml.data import validate
from finml.monitoring import OUTPUT_COLUMNS, PredictionLogger
from finml.train import default_tracking_uri

log = logging.getLogger("finml.api")

# Métriques Prometheus, étiquetées par cas d'usage. Un seul worker par conteneur :
# la montée en charge se fait en ajoutant des tâches ECS, chaque tâche expose ses compteurs.
PREDICTIONS = Counter("finml_predictions_total", "Lignes prédites", ["usecase"])
ERRORS = Counter("finml_prediction_errors_total", "Requêtes en erreur", ["usecase", "raison"])
LATENCY = Histogram("finml_prediction_latency_seconds", "Latence d'une requête de prédiction", ["usecase"])
SCORES = Histogram(
    "finml_prediction_score", "Distribution du score de sortie (probabilité ou confiance)", ["usecase"],
    buckets=[i / 10 for i in range(1, 11)],
)
LLM_CALLS = Counter("finml_llm_second_avis_total", "Appels au second avis LLM", ["usecase", "resultat"])
MODEL_INFO = Gauge("finml_model_info", "Version du modèle servi", ["usecase", "version", "candidat"])


class SecondAvisRequest(BaseModel):
    libelle: str = Field(min_length=1, max_length=500)


class PredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1, max_length=1000)


class LoadedModel:
    def __init__(self, cfg: UseCaseConfig, model, source: str, manifest: dict):
        self.cfg, self.model, self.source, self.manifest = cfg, model, source, manifest


def _load_one(name: str, models_dir: Path) -> LoadedModel:
    cfg = load_config(name)
    local = models_dir / name
    if (local / "MLmodel").exists():
        manifest_file = local / "manifest.json"
        manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
        return LoadedModel(cfg, mlflow.pyfunc.load_model(str(local)), str(local), manifest)
    uri = f"models:/{cfg.registered_model}@champion"
    return LoadedModel(cfg, mlflow.pyfunc.load_model(uri), uri, {"alias": "champion"})


def create_app(models_dir: str | Path | None = None, usecases: list[str] | None = None,
               prediction_log_dir: str | Path | None = None) -> FastAPI:
    models_dir = Path(models_dir or os.environ.get("FINML_MODELS_DIR", "models"))
    mlflow.set_tracking_uri(default_tracking_uri())
    pred_log = PredictionLogger(prediction_log_dir)
    registry: dict[str, LoadedModel] = {}
    errors: dict[str, str] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        for name in usecases or list_usecases():
            try:
                registry[name] = _load_one(name, models_dir)
                m = registry[name]
                MODEL_INFO.labels(name, str(m.manifest.get("version")), str(m.manifest.get("candidate"))).set(1)
                log.info("modèle chargé : %s depuis %s", name, m.source)
            except Exception as e:  # noqa: BLE001 - le service démarre, /health signale le problème
                errors[name] = f"{type(e).__name__}: {e}"
                log.error("échec de chargement %s : %s", name, e)
        yield

    app = FastAPI(
        title="finml - moteur de prédiction financière",
        description="Un socle MLOps, plusieurs cas d'usage déclarés en configuration.",
        version="0.2.0",
        lifespan=lifespan,
    )
    app.mount("/metrics", make_asgi_app())

    @app.get("/health")
    def health():
        body = {"status": "ok" if registry and not errors else "degraded",
                "modeles": sorted(registry), "erreurs": errors}
        return JSONResponse(body, status_code=200 if body["status"] == "ok" else 503)

    @app.get("/usecases")
    def get_usecases():
        return [
            {
                "nom": name,
                "description": m.cfg.description,
                "type": m.cfg.task,
                "entrees": m.cfg.features.all,
                "version": m.manifest.get("version"),
                "candidat": m.manifest.get("candidate"),
                "metrique": m.manifest.get("metric", m.cfg.metric),
                "score": m.manifest.get("score"),
                "source": m.source,
            }
            for name, m in sorted(registry.items())
        ]

    @app.post("/predict/{usecase}")
    def predict(usecase: str, req: PredictRequest):
        if usecase not in registry:
            ERRORS.labels(usecase, "cas_inconnu").inc()
            raise HTTPException(404, f"Cas d'usage inconnu ou non chargé : {usecase}")
        m = registry[usecase]
        t0 = time.perf_counter()
        df = pd.DataFrame(req.records)
        report = validate(df, m.cfg, with_target=False)
        if not report.ok:
            ERRORS.labels(usecase, "donnees_invalides").inc()
            raise HTTPException(422, {"erreurs_donnees": report.errors})
        X = df[m.cfg.features.all].astype({c: "float64" for c in m.cfg.features.numeric})
        try:
            out = m.model.predict(X)
        except Exception as e:
            ERRORS.labels(usecase, "erreur_modele").inc()
            log.exception("échec de prédiction %s", usecase)
            raise HTTPException(500, "Échec de la prédiction") from e
        latency = time.perf_counter() - t0
        records = out.to_dict(orient="records")

        LATENCY.labels(usecase).observe(latency)
        PREDICTIONS.labels(usecase).inc(len(records))
        score_col = next((c for c in OUTPUT_COLUMNS if c in out), None)
        if score_col:
            for v in out[score_col]:
                SCORES.labels(usecase).observe(float(v))
        try:
            pred_log.write(usecase, m.manifest.get("version"), X.to_dict(orient="records"), records,
                           round(latency * 1000, 1))
        except OSError:
            log.exception("journal des prédictions indisponible")  # ne bloque jamais la réponse

        for r in records:
            for key in ("top3", "exemples_proches"):
                if isinstance(r.get(key), str):
                    r[key] = json.loads(r[key])
        return {
            "usecase": usecase,
            "modele_version": m.manifest.get("version"),
            "latence_ms": round(latency * 1000, 1),
            "predictions": records,
        }

    @app.post("/second-avis/{usecase}")
    def second_avis_endpoint(usecase: str, req: SecondAvisRequest):
        """Proposition du modèle, complétée par un second avis LLM si la pièce part en revue."""
        from finml.llm import LLMNotConfigured, make_client, second_avis

        if usecase not in registry:
            raise HTTPException(404, f"Cas d'usage inconnu ou non chargé : {usecase}")
        m = registry[usecase]
        if m.cfg.task != "text_classification":
            raise HTTPException(400, "Second avis réservé aux cas d'usage texte")
        try:
            client = make_client(m.cfg, os.environ.get("FINML_LLM_PROVIDER", "auto"))
        except LLMNotConfigured as e:
            raise HTTPException(503, str(e)) from e
        out = m.model.predict(pd.DataFrame({m.cfg.features.text: [req.libelle]})).iloc[0].to_dict()
        top3, exemples = json.loads(out["top3"]), json.loads(out["exemples_proches"])
        avis = None
        if out["a_revoir"]:
            result = second_avis(m.cfg, client, req.libelle, top3, exemples)
            avis = result.to_dict()
            LLM_CALLS.labels(usecase, "valide" if result.valide else (result.rejet or "rejet")).inc()
        return {"usecase": usecase, "modele": {**out, "top3": top3, "exemples_proches": exemples},
                "second_avis": avis}

    return app


app = create_app()

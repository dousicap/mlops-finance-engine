"""Entraînement : tous les candidats, suivi MLflow, sélection, porte qualité, Registry et alias.

Règles de promotion :
- le meilleur candidat doit atteindre `min_score`, sinon rien n'est enregistré (la CI échoue) ;
- la nouvelle version reçoit l'alias `champion` seulement si elle bat le champion actuel
  sur la métrique déclarée, sinon elle reçoit l'alias `challenger`.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import finml
from finml import data as data_mod
from finml.config import UseCaseConfig, load_config
from finml.models import build_estimator, dump_estimator, predict_frame, text_class_scores

CHAMPION, CHALLENGER = "champion", "challenger"
OUTPUT_COLUMNS = ("probabilite", "confiance")


class QualityGateError(RuntimeError):
    """Le meilleur modèle n'atteint pas le score minimal déclaré."""


@dataclass
class TrainResult:
    usecase: str
    best_candidate: str
    score: float
    metric: str
    run_id: str
    version: str
    alias: str
    all_scores: dict[str, float]


def default_tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{Path.cwd() / 'mlflow.db'}")


def evaluate(cfg: UseCaseConfig, est, X_test, y_test) -> dict[str, float]:
    if cfg.task == "tabular_classification":
        proba = est.predict_proba(X_test)[:, 1]
        pred = (proba >= cfg.decision_threshold).astype(int)
        return {
            "roc_auc": roc_auc_score(y_test, proba),
            "average_precision": average_precision_score(y_test, proba),
            "f1": f1_score(y_test, pred),
            "accuracy": accuracy_score(y_test, pred),
        }
    classes, scores = text_class_scores(est, X_test)
    pred = classes[scores.argmax(axis=1)]
    top3 = classes[np.argsort(-scores, axis=1)[:, :3]]
    y = np.asarray(y_test)
    return {
        "accuracy": accuracy_score(y, pred),
        "f1": f1_score(y, pred, average="macro"),
        "top3_accuracy": float(np.mean([y[i] in top3[i] for i in range(len(y))])),
    }


def _report(cfg: UseCaseConfig, est, X_test, y_test) -> dict:
    if cfg.task == "tabular_classification":
        pred = (est.predict_proba(X_test)[:, 1] >= cfg.decision_threshold).astype(int)
    else:
        classes, scores = text_class_scores(est, X_test)
        pred = classes[scores.argmax(axis=1)]
    return classification_report(y_test, pred, output_dict=True, zero_division=0)


def _promote(client: MlflowClient, cfg: UseCaseConfig, version: str, score: float) -> str:
    try:
        current = client.get_model_version_by_alias(cfg.registered_model, CHAMPION)
        current_score = client.get_run(current.run_id).data.metrics.get(cfg.metric, -1.0)
    except MlflowException:
        current_score = None
    if current_score is None or score > current_score:
        client.set_registered_model_alias(cfg.registered_model, CHAMPION, version)
        return CHAMPION
    client.set_registered_model_alias(cfg.registered_model, CHALLENGER, version)
    return CHALLENGER


def train(name: str) -> TrainResult:
    cfg = load_config(name)
    df = data_mod.load_validated(cfg)
    mlflow.set_tracking_uri(default_tracking_uri())
    mlflow.set_experiment(f"finml-{cfg.name}")

    if cfg.task == "tabular_classification":
        X = df[cfg.features.all].astype({c: "float64" for c in cfg.features.numeric})
        y = df[cfg.target]
    else:
        X, y = df[cfg.features.text].astype(str), df[cfg.target].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.data.seed, stratify=y
    )

    scores: dict[str, float] = {}
    runs: dict[str, str] = {}
    model_uris: dict[str, str] = {}
    with mlflow.start_run(run_name=f"selection-{time.strftime('%Y%m%d-%H%M%S')}"):
        mlflow.set_tags(
            {
                "usecase": cfg.name,
                "data_sha256": data_mod.fingerprint(df),
                "git_sha": os.environ.get("GITHUB_SHA", "local"),
                "finml_version": finml.__version__,
            }
        )
        mlflow.log_params({"n_rows": len(df), "test_size": cfg.test_size, "metric": cfg.metric})

        for cand in cfg.candidates:
            with mlflow.start_run(run_name=cand.name, nested=True) as run:
                est = build_estimator(cfg, cand)
                t0 = time.perf_counter()
                est.fit(X_train, y_train)
                metrics = evaluate(cfg, est, X_test, y_test)
                metrics["train_seconds"] = time.perf_counter() - t0
                mlflow.log_params({"model_type": cand.type, **{f"p_{k}": v for k, v in cand.params.items()}})
                mlflow.log_metrics(metrics)
                with tempfile.TemporaryDirectory() as tmp:
                    rep = Path(tmp) / "classification_report.json"
                    rep.write_text(json.dumps(_report(cfg, est, X_test, y_test), indent=2))
                    mlflow.log_artifact(str(rep))
                example = X_test.head(3).to_frame() if cfg.task == "text_classification" else X_test.head(3)
                model_info = _log_model(cfg, est, X_test, example)
                scores[cand.name] = float(metrics[cfg.metric])
                runs[cand.name] = run.info.run_id
                model_uris[cand.name] = model_info.model_uri

        best = max(scores, key=scores.get)
        mlflow.log_metric(f"best_{cfg.metric}", scores[best])
        mlflow.set_tag("best_candidate", best)
        if scores[best] < cfg.min_score:
            mlflow.set_tag("quality_gate", "failed")
            raise QualityGateError(
                f"{cfg.name} : meilleur {cfg.metric} = {scores[best]:.3f} < seuil {cfg.min_score}"
            )
        mlflow.set_tag("quality_gate", "passed")

    mv = mlflow.register_model(model_uris[best], cfg.registered_model)
    client = MlflowClient()
    client.set_model_version_tag(cfg.registered_model, mv.version, cfg.metric, f"{scores[best]:.4f}")
    alias = _promote(client, cfg, mv.version, scores[best])
    return TrainResult(cfg.name, best, scores[best], cfg.metric, runs[best], mv.version, alias, scores)


SERVING_CODE = Path(__file__).with_name("serving_model.py")
REFERENCE_ROWS = 2000


def _log_model(cfg: UseCaseConfig, est, X_test, example):
    """Journalise le modèle sans pickle : code source + estimateur skops + configuration JSON.

    Un échantillon de référence (entrées du jeu de test et sorties du modèle) est joint :
    il sert à mesurer la dérive en production, sans dépendre des données d'entraînement.
    """
    X_ref = X_test.to_frame() if cfg.task == "text_classification" else X_test
    X_ref = X_ref.head(REFERENCE_ROWS).reset_index(drop=True)
    outputs = predict_frame(est, cfg, X_ref)
    reference = pd.concat([X_ref, outputs[[c for c in OUTPUT_COLUMNS if c in outputs]]], axis=1)
    with tempfile.TemporaryDirectory() as tmp:
        est_path, cfg_path, ref_path = Path(tmp) / "estimator.skops", Path(tmp) / "config.json", Path(tmp) / "reference.csv"
        dump_estimator(est, est_path)
        cfg_path.write_text(cfg.model_dump_json(by_alias=True), encoding="utf-8")
        reference.to_csv(ref_path, index=False)
        return mlflow.pyfunc.log_model(
            name="model",
            python_model=str(SERVING_CODE),
            artifacts={"estimator": str(est_path), "config": str(cfg_path), "reference": str(ref_path)},
            code_paths=[str(Path(finml.__file__).parent)],
            signature=infer_signature(example),
            input_example=example,
            pip_requirements=_requirements(),
        )


def _requirements() -> list[str]:
    import pandas
    import sklearn
    import skops
    import xgboost

    return [
        f"mlflow=={mlflow.__version__}",
        f"scikit-learn=={sklearn.__version__}",
        f"xgboost-cpu=={xgboost.__version__}",
        f"pandas=={pandas.__version__}",
        f"skops=={skops.__version__}",
        "pydantic>=2.7,<3",
        "pyyaml>=6.0",
    ]

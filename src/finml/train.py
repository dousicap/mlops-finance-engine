"""Entraînement : tous les candidats, suivi MLflow, sélection, porte qualité, Registry et alias.

Règles de promotion :
- le meilleur candidat doit atteindre `min_score`, sinon rien n'est enregistré (la CI échoue) ;
- la nouvelle version reçoit l'alias `champion` seulement si elle bat le champion actuel
  sur la métrique déclarée, sinon elle reçoit l'alias `challenger`.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field
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
from sklearn.model_selection import StratifiedKFold, train_test_split

import finml
from finml import business
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
    leaderboard: list[dict] = field(default_factory=list)
    business_metrics: dict[str, float] = field(default_factory=dict)


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


def _scores(cfg: UseCaseConfig, est, X):
    """Tabulaire : probabilité de la classe positive. Texte : (classe prédite, confiance)."""
    if cfg.task == "tabular_classification":
        return est.predict_proba(X)[:, 1]
    classes, scores = text_class_scores(est, X)
    return classes[scores.argmax(axis=1)], scores.max(axis=1)


OPTIONAL_DEPS = {"torch_mlp": "torch"}


def _available(cand) -> bool:
    """Un candidat dont la bibliothèque optionnelle manque est ignoré, pas bloquant."""
    import importlib.util

    module = OPTIONAL_DEPS.get(cand.type)
    return module is None or importlib.util.find_spec(module) is not None


def _cross_validate(cfg: UseCaseConfig, cand, X, y):
    """Validation croisée stratifiée : score par pli et prédictions hors échantillon."""
    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.data.seed)
    fold_scores = []
    oof_proba = np.zeros(len(X)) if cfg.task == "tabular_classification" else None
    oof_pred = np.empty(len(X), dtype=object) if oof_proba is None else None
    oof_conf = np.zeros(len(X)) if oof_proba is None else None
    for tr, va in skf.split(X, y):
        est = build_estimator(cfg, cand).fit(X.iloc[tr], y.iloc[tr])
        fold_scores.append(evaluate(cfg, est, X.iloc[va], y.iloc[va])[cfg.metric])
        out = _scores(cfg, est, X.iloc[va])
        if oof_proba is not None:
            oof_proba[va] = out
        else:
            oof_pred[va], oof_conf[va] = out
    return np.array(fold_scores), (oof_proba if oof_proba is not None else (oof_pred, oof_conf))


def _calibrate_business(cfg: UseCaseConfig, y_train, oof, est, X_test, y_test):
    """Choisit le seuil sur les prédictions hors échantillon, le mesure sur le jeu de test."""
    b = cfg.business
    if b is None:
        return cfg.decision_threshold, {}
    if b.kind == "cost":
        threshold = business.best_cost_threshold(y_train, oof, b)
        return threshold, business.cost_report(y_test, _scores(cfg, est, X_test), threshold, b)
    pred, conf = oof
    threshold = business.best_automation_threshold(y_train, pred, conf, b)
    test_pred, test_conf = _scores(cfg, est, X_test)
    return threshold, business.automation_report(y_test, test_pred, test_conf, threshold, b)


def train(name: str) -> TrainResult:
    """Tournoi : chaque candidat passe la validation croisée, le meilleur en moyenne devient
    le candidat retenu, il est réentraîné sur tout le jeu d'entraînement puis mesuré sur le
    jeu de test que personne n'a vu pendant la sélection."""
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
    X_train, X_test = X_train.reset_index(drop=True), X_test.reset_index(drop=True)
    y_train, y_test = y_train.reset_index(drop=True), y_test.reset_index(drop=True)

    board, runs, fitted, oofs = [], {}, {}, {}
    with mlflow.start_run(run_name=f"tournoi-{time.strftime('%Y%m%d-%H%M%S')}"):
        mlflow.set_tags(
            {
                "usecase": cfg.name,
                "data_sha256": data_mod.fingerprint(df),
                "git_sha": os.environ.get("GITHUB_SHA", "local"),
                "finml_version": finml.__version__,
            }
        )
        mlflow.log_params({"n_rows": len(df), "test_size": cfg.test_size, "metric": cfg.metric,
                           "cv_folds": cfg.cv_folds, "n_candidates": len(cfg.candidates)})

        skipped: list[str] = []
        for cand in cfg.candidates:
            if not _available(cand):
                skipped.append(cand.name)  # dépendance optionnelle absente (ex. PyTorch)
                continue
            with mlflow.start_run(run_name=cand.name, nested=True) as run:
                t0 = time.perf_counter()
                cv, oof = _cross_validate(cfg, cand, X_train, y_train)
                est = build_estimator(cfg, cand).fit(X_train, y_train)
                holdout = evaluate(cfg, est, X_test, y_test)
                seconds = time.perf_counter() - t0
                mlflow.log_params({"model_type": cand.type, **{f"p_{k}": v for k, v in cand.params.items()}})
                mlflow.log_metrics({**holdout, f"cv_{cfg.metric}_mean": float(cv.mean()),
                                    f"cv_{cfg.metric}_std": float(cv.std()), "train_seconds": seconds})
                board.append({"candidat": cand.name, "type": cand.type, "cv_moyenne": float(cv.mean()),
                              "cv_ecart_type": float(cv.std()), "test": float(holdout[cfg.metric]),
                              "secondes": round(seconds, 1)})
                runs[cand.name], fitted[cand.name], oofs[cand.name] = run.info.run_id, est, oof

        if skipped:
            mlflow.set_tag("candidats_ignores", ",".join(skipped))
        # Sélection sur la moyenne en validation croisée, jamais sur le jeu de test.
        board.sort(key=lambda r: r["cv_moyenne"], reverse=True)
        best = board[0]["candidat"]
        best_test = board[0]["test"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "classement.csv"
            pd.DataFrame(board).to_csv(path, index=False)
            mlflow.log_artifact(str(path))
        mlflow.set_tag("best_candidate", best)
        mlflow.log_metric(f"best_{cfg.metric}", best_test)
        if best_test < cfg.min_score:
            mlflow.set_tag("quality_gate", "failed")
            raise QualityGateError(f"{cfg.name} : meilleur {cfg.metric} = {best_test:.3f} < seuil {cfg.min_score}")
        mlflow.set_tag("quality_gate", "passed")

        threshold, biz = _calibrate_business(cfg, y_train, oofs[best], fitted[best], X_test, y_test)
        served_cfg = cfg.model_copy(update={"decision_threshold": threshold})
        with mlflow.start_run(run_id=runs[best], nested=True):
            mlflow.log_metrics(biz)
            example = X_test.head(3).to_frame() if cfg.task == "text_classification" else X_test.head(3)
            model_info = _log_model(served_cfg, fitted[best], X_test, example)
        mlflow.log_metrics(biz)

    mv = mlflow.register_model(model_info.model_uri, cfg.registered_model)
    client = MlflowClient()
    client.set_model_version_tag(cfg.registered_model, mv.version, cfg.metric, f"{best_test:.4f}")
    alias = _promote(client, cfg, mv.version, best_test)
    scores = {r["candidat"]: r["test"] for r in board}
    return TrainResult(cfg.name, best, best_test, cfg.metric, runs[best], mv.version, alias, scores,
                       board, biz)


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

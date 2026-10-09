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
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

import finml
from finml import business
from finml import data as data_mod
from finml.config import LOWER_IS_BETTER, Candidate, UseCaseConfig, load_config
from finml.evaluation import compare_to_champion, evaluate_champion, leakage_check
from finml.fairness import fairness_report
from finml.models import (
    BASELINE,
    build_estimator,
    class_scores,
    dump_estimator,
    is_binary,
    predict_frame,
)
from finml.tuning import pick_champion, tune

CHAMPION, CHALLENGER = "champion", "challenger"
OUTPUT_COLUMNS = ("probabilite", "confiance", "valeur")


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
    fairness: dict = field(default_factory=dict)
    evaluation: dict[str, float] = field(default_factory=dict)
    selection_note: str | None = None


def default_tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{Path.cwd() / 'mlflow.db'}")


def evaluate(cfg: UseCaseConfig, est, X_test, y_test) -> dict[str, float]:
    if cfg.is_regression:
        pred = est.predict(X_test)
        return {
            "rmse": float(root_mean_squared_error(y_test, pred)),
            "mae": float(mean_absolute_error(y_test, pred)),
            "r2": float(r2_score(y_test, pred)),
        }
    if cfg.is_tabular and is_binary(est):
        proba = est.predict_proba(X_test)[:, 1]
        pred = (proba >= cfg.decision_threshold).astype(int)
        return {
            "roc_auc": roc_auc_score(y_test, proba),
            "average_precision": average_precision_score(y_test, proba),
            "f1": f1_score(y_test, pred),
            "accuracy": accuracy_score(y_test, pred),
        }
    classes, scores = class_scores(est, cfg, X_test)
    pred = classes[scores.argmax(axis=1)]
    top3 = classes[np.argsort(-scores, axis=1)[:, :3]]
    y = np.asarray(y_test)
    metrics = {
        "accuracy": accuracy_score(y, pred),
        "f1": f1_score(y, pred, average="macro"),
        "top3_accuracy": float(np.mean([y[i] in top3[i] for i in range(len(y))])),
    }
    if cfg.is_tabular:  # AUC multi-classes : une classe contre les autres, moyenne pondérée
        metrics["roc_auc"] = roc_auc_score(y, scores, multi_class="ovr", average="weighted", labels=classes)
    return metrics


def _report(cfg: UseCaseConfig, est, X_test, y_test) -> dict:
    if cfg.is_tabular and is_binary(est):
        pred = (est.predict_proba(X_test)[:, 1] >= cfg.decision_threshold).astype(int)
    else:
        classes, scores = class_scores(est, cfg, X_test)
        pred = classes[scores.argmax(axis=1)]
    return classification_report(y_test, pred, output_dict=True, zero_division=0)


def _promote(client: MlflowClient, cfg: UseCaseConfig, version: str, score: float) -> str:
    try:
        current = client.get_model_version_by_alias(cfg.registered_model, CHAMPION)
        current_score = client.get_run(current.run_id).data.metrics.get(cfg.metric, -1.0)
    except MlflowException:
        current_score = None
    if current_score is None or cfg.better(score, current_score):
        client.set_registered_model_alias(cfg.registered_model, CHAMPION, version)
        return CHAMPION
    client.set_registered_model_alias(cfg.registered_model, CHALLENGER, version)
    return CHALLENGER


def _scores(cfg: UseCaseConfig, est, X):
    """Binaire : probabilité de la classe positive. Multi-classes : (classe prédite, confiance).
    Régression : valeur prédite."""
    if cfg.is_regression:
        return est.predict(X)
    if cfg.is_tabular and is_binary(est):
        return est.predict_proba(X)[:, 1]
    classes, scores = class_scores(est, cfg, X)
    return classes[scores.argmax(axis=1)], scores.max(axis=1)


OPTIONAL_DEPS = {"torch_mlp": "torch"}
BINARY_ONLY = {"torch_mlp"}


def _skip_reason(cand, n_classes: int) -> str | None:
    """Un candidat inutilisable est ignoré avec sa raison, jamais bloquant."""
    import importlib.util

    module = OPTIONAL_DEPS.get(cand.type)
    if module and importlib.util.find_spec(module) is None:
        return f"bibliothèque {module} absente"
    if cand.type in BINARY_ONLY and n_classes > 2:
        return "modèle limité à la classification binaire"
    return None


def _folds(cfg: UseCaseConfig, X, y, offset: int = 0):
    """Plis de validation croisée, identiques pour tous les candidats.

    offset=1 : autre mélange, réservé à la recherche d'hyperparamètres. Le réglage est ensuite
    jugé sur les plis du classement, qu'il n'a pas vus : le meilleur de N essais mesurés sur les
    mêmes plis serait un score flatteur (biais de sélection).
    """
    splitter = KFold if cfg.is_regression else StratifiedKFold
    return splitter(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.data.seed + offset).split(X, y)


def _fold_score(cfg: UseCaseConfig, cand, X, y, tr, va) -> float:
    est = build_estimator(cfg, cand).fit(X.iloc[tr], y.iloc[tr])
    return evaluate(cfg, est, X.iloc[va], y.iloc[va])[cfg.metric]


def _clearly_better(cfg: UseCaseConfig, tuned_scores, start_scores) -> bool:
    """Le réglage optimisé bat-il nettement le réglage de départ, sur les mêmes plis ?

    Même règle d'égalité que pour le choix du champion : sans écart significatif (test t
    corrigé) et sous la tolérance, les deux sont à égalité et le réglage de départ est gardé.
    """
    from finml.evaluation import corrected_t_test

    gain = float(np.mean(tuned_scores) - np.mean(start_scores))
    if not cfg.better(float(np.mean(tuned_scores)), float(np.mean(start_scores))):
        return False
    p_value = corrected_t_test(tuned_scores, start_scores, 1 / (cfg.cv_folds - 1))
    return p_value <= cfg.selection.alpha or abs(gain) > cfg.selection.tolerance


def _cross_validate(cfg: UseCaseConfig, cand, X, y):
    """Validation croisée stratifiée : score par pli et prédictions hors échantillon."""
    fold_scores = []
    single = cfg.is_regression or (cfg.is_tabular and y.nunique() == 2)  # une sortie numérique
    oof_proba = np.zeros(len(X)) if single else None
    oof_pred = np.empty(len(X), dtype=object) if oof_proba is None else None
    oof_conf = np.zeros(len(X)) if oof_proba is None else None
    for tr, va in _folds(cfg, X, y):
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
        if y_train.nunique() != 2:
            raise ValueError(f"{cfg.name} : le bloc business « cost » exige une cible binaire")
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

    positive = None
    if cfg.is_tabular:
        X = df[cfg.features.all].astype({c: "float64" for c in cfg.features.numeric})
        y = df[cfg.target]
        if not cfg.is_regression and y.nunique() == 2 and set(y.unique()) != {0, 1}:
            # Cible binaire quelconque (oui/non, défaut/sain) : codée 0/1, la classe positive
            # est positive_label si déclarée, sinon la dernière dans l'ordre alphabétique.
            positive = cfg.positive_label if cfg.positive_label is not None else max(y.unique(), key=str)
            negative = next(v for v in y.unique() if v != positive)
            y = (y == positive).astype(int)
            cfg = cfg.model_copy(update={"labels": {"1": str(positive), "0": str(negative)}})
    else:
        X, y = df[cfg.features.text].astype(str), df[cfg.target].astype(str)
    # Les colonnes sensibles suivent le même découpage que X : elles ne servent qu'à l'audit.
    S = df[cfg.fairness.sensitive] if cfg.fairness else pd.DataFrame(index=df.index)
    X_train, X_test, y_train, y_test, _, S_test = train_test_split(
        X, y, S, test_size=cfg.test_size, random_state=cfg.data.seed,
        stratify=None if cfg.is_regression else y,
    )
    n_classes = 0 if cfg.is_regression else int(y.nunique())
    X_train, X_test = X_train.reset_index(drop=True), X_test.reset_index(drop=True)
    y_train, y_test = y_train.reset_index(drop=True), y_test.reset_index(drop=True)

    board, runs, fitted, oofs, fold_scores = [], {}, {}, {}, {}
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

        leaks = leakage_check(cfg, X_train, y_train) if cfg.is_tabular else {}
        if leaks:  # une seule variable prédit presque parfaitement la cible : fuite probable
            mlflow.set_tag("fuite_suspectee", ", ".join(f"{c} ({s:.3f})" for c, s in leaks.items()))
        skipped: list[str] = []
        # Référence naïve toujours en lice : un champion qui ne la bat pas ne sert à rien.
        baseline = Candidate.model_construct(name=BASELINE, type="dummy", params={})
        for cand in [baseline, *cfg.candidates]:
            reason = _skip_reason(cand, n_classes)
            if reason:
                skipped.append(f"{cand.name} ({reason})")
                continue
            with mlflow.start_run(run_name=cand.name, nested=True) as run:
                t0 = time.perf_counter()
                cv, oof = _cross_validate(cfg, cand, X_train, y_train)
                cv_start, n_trials, tuned = float(cv.mean()), 0, False
                if cfg.tuning and cand.search:
                    params, trials = tune(cfg, cand, X_train, y_train, lambda X, y: _folds(cfg, X, y, offset=1),
                                          lambda c, X, y, tr, va: _fold_score(cfg, c, X, y, tr, va))
                    n_trials = len(trials)
                    if params:
                        tuned_cand = cand.model_copy(update={"params": params})
                        cv_t, oof_t = _cross_validate(cfg, tuned_cand, X_train, y_train)
                        if _clearly_better(cfg, cv_t, cv):  # à égalité, le réglage de départ reste
                            cand, cv, oof, tuned = tuned_cand, cv_t, oof_t, True
                fold_scores[cand.name] = cv
                est = build_estimator(cfg, cand).fit(X_train, y_train)
                holdout = evaluate(cfg, est, X_test, y_test)
                seconds = time.perf_counter() - t0
                mlflow.log_params({"model_type": cand.type, **{f"p_{k}": v for k, v in cand.params.items()}})
                mlflow.log_metrics({**holdout, f"cv_{cfg.metric}_mean": float(cv.mean()),
                                    f"cv_{cfg.metric}_std": float(cv.std()), "train_seconds": seconds})
                board.append({"candidat": cand.name, "type": cand.type, "cv_moyenne": float(cv.mean()),
                              "cv_ecart_type": float(cv.std()), "test": float(holdout[cfg.metric]),
                              "secondes": round(seconds, 1), "cv_depart": cv_start, "essais": n_trials,
                              "reglage": "optuna" if tuned else "depart"})
                if n_trials:
                    mlflow.set_tags({"reglage": "optuna" if tuned else "depart", "essais": n_trials})
                runs[cand.name], fitted[cand.name], oofs[cand.name] = run.info.run_id, est, oof

        if skipped:
            mlflow.set_tag("candidats_ignores", ", ".join(skipped))
        # Sélection sur la moyenne en validation croisée, jamais sur le jeu de test.
        board.sort(key=lambda r: r["cv_moyenne"], reverse=cfg.metric not in LOWER_IS_BETTER)
        compare_to_champion(board, fold_scores, cfg)  # écart significatif ou égalité statistique
        if board[0]["candidat"] == BASELINE:
            mlflow.set_tag("quality_gate", "failed")
            raise QualityGateError(f"{cfg.name} : aucun candidat ne bat la référence naïve")
        best, note = pick_champion(cfg, board)  # à égalité, le plus simple gagne
        if note:
            mlflow.set_tag("regle_simplicite", note)
            compare_to_champion(board, fold_scores, cfg, champion=best)
        best_test = next(r["test"] for r in board if r["candidat"] == best)
        if positive is not None:
            mlflow.set_tag("classe_positive", str(positive))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "classement.csv"
            pd.DataFrame(board).to_csv(path, index=False)
            mlflow.log_artifact(str(path))
        mlflow.set_tag("best_candidate", best)
        mlflow.log_metric(f"best_{cfg.metric}", best_test)
        if not cfg.passes_gate(best_test):
            mlflow.set_tag("quality_gate", "failed")
            limit = cfg.max_score if cfg.min_score is None else cfg.min_score
            raise QualityGateError(f"{cfg.name} : meilleur {cfg.metric} = {best_test:.3f}, seuil {limit}")
        mlflow.set_tag("quality_gate", "passed")

        threshold, biz = _calibrate_business(cfg, y_train, oofs[best], fitted[best], X_test, y_test)
        served_cfg = cfg.model_copy(update={"decision_threshold": threshold})
        fair_metrics, fair_detail = {}, {}
        if cfg.fairness:
            if n_classes != 2:
                raise ValueError(f"{cfg.name} : l'audit d'équité exige une cible binaire")
            flag = _scores(cfg, fitted[best], X_test) >= threshold
            fair_metrics, fair_detail = fairness_report(y_test, flag, S_test, cfg.fairness)
            mlflow.set_tag("equite_statut", fair_detail["statut"])
            mlflow.log_dict(fair_detail, "equite.json")
        _log_charts(cfg, board, best, fitted[best], threshold, X_test, y_test)
        eval_metrics, eval_detail = evaluate_champion(cfg, fitted[best], threshold, X_test, y_test)
        mlflow.log_dict(eval_detail, "evaluation.json")
        with mlflow.start_run(run_id=runs[best], nested=True):
            mlflow.log_metrics({**biz, **fair_metrics, **eval_metrics})
            example = X_test.head(3).to_frame() if cfg.task == "text_classification" else X_test.head(3)
            model_info = _log_model(served_cfg, fitted[best], X_test, example)
        mlflow.log_metrics({**biz, **fair_metrics, **eval_metrics})

    mv = mlflow.register_model(model_info.model_uri, cfg.registered_model)
    client = MlflowClient()
    client.set_model_version_tag(cfg.registered_model, mv.version, cfg.metric, f"{best_test:.4f}")
    alias = _promote(client, cfg, mv.version, best_test)
    scores = {r["candidat"]: r["test"] for r in board}
    return TrainResult(cfg.name, best, best_test, cfg.metric, runs[best], mv.version, alias, scores,
                       board, biz, fair_detail, eval_metrics, note)


def _log_charts(cfg: UseCaseConfig, board: list[dict], champion: str, est, threshold: float, X_test,
                y_test) -> None:
    """Classement du tournoi et diagnostics du champion, en images dans MLflow."""
    import matplotlib.pyplot as plt

    from finml import plots

    figures = {"classement_tournoi": plots.leaderboard(board, cfg.metric, champion)}
    if cfg.is_regression:
        figures["predit_contre_reel"] = plots.regression(y_test, est.predict(X_test))
    elif cfg.is_tabular and is_binary(est):
        proba = est.predict_proba(X_test)[:, 1]
        figures["matrice_confusion"] = plots.confusion(y_test, (proba >= threshold).astype(int), [0, 1])
        figures["courbes_roc_precision_rappel"] = plots.binary_curves(y_test, proba)
        figures["calibration"] = plots.calibration(y_test, proba)
        figures["metriques_selon_seuil"] = plots.threshold_metrics(y_test, proba, threshold)
    else:
        classes, scores = class_scores(est, cfg, X_test)
        figures["matrice_confusion"] = plots.confusion(np.asarray(y_test), classes[scores.argmax(axis=1)],
                                                       list(classes))
    for name, fig in figures.items():
        mlflow.log_figure(fig, f"graphiques/{name}.png")
        plt.close(fig)


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

"""Recherche d'hyperparamètres par candidat (Optuna).

- Espace de recherche déclaré dans config.yaml (`search` de chaque candidat), budget dans le
  bloc `tuning` : nombre d'essais et durée maximale par candidat. FINML_TUNING_TRIALS remplace
  le nombre d'essais (budget réduit en CI).
- Échantillonneur TPE à graine fixe : deux exécutions donnent les mêmes essais.
- Chaque essai est une validation croisée sur des plis de recherche (autre mélange que le
  classement). Après chaque pli, l'essai est élagué s'il est sous la médiane des essais
  précédents au même pli.
- Chaque essai est tracé dans MLflow (run imbriqué), la table complète en artefact.
- Le réglage trouvé est ensuite mesuré sur les plis du classement, qu'il n'a pas vus, et n'est
  retenu que s'il y bat le réglage de départ. Le réglage ne peut donc jamais dégrader la
  validation croisée d'un candidat, et son score n'est pas gonflé par la sélection.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import mlflow
import numpy as np

from finml.config import LOWER_IS_BETTER, Candidate, UseCaseConfig

# Plages par défaut, écrites par `finml init` dans la configuration générée.
DEFAULT_SPACES: dict[str, dict] = {
    "logistic_regression": {"C": {"float": [0.01, 100.0], "log": True}},
    "svm": {"C": {"float": [0.01, 10.0], "log": True}},
    "naive_bayes": {"var_smoothing": {"float": [1e-12, 1e-6], "log": True}},
    "knn": {"n_neighbors": {"int": [5, 75]}},
    "decision_tree": {"max_depth": {"int": [2, 12]}, "min_samples_leaf": {"int": [1, 50]}},
    "random_forest": {"max_depth": {"int": [4, 20]}, "min_samples_leaf": {"int": [1, 30]},
                      "max_features": {"choice": ["sqrt", 0.3, 0.6]}},
    "extra_trees": {"max_depth": {"int": [4, 20]}, "min_samples_leaf": {"int": [1, 30]},
                    "max_features": {"choice": ["sqrt", 0.3, 0.6]}},
    "hist_gradient_boosting": {"learning_rate": {"float": [0.02, 0.3], "log": True},
                               "max_depth": {"int": [2, 10]}, "min_samples_leaf": {"int": [10, 100]},
                               "l2_regularization": {"float": [1e-4, 10.0], "log": True}},
    "xgboost": {"n_estimators": {"int": [100, 800]}, "max_depth": {"int": [2, 8]},
                "learning_rate": {"float": [0.01, 0.3], "log": True}, "subsample": {"float": [0.6, 1.0]},
                "colsample_bytree": {"float": [0.5, 1.0]}, "min_child_weight": {"int": [1, 20]}},
    "lightgbm": {"n_estimators": {"int": [100, 800]}, "num_leaves": {"int": [7, 63]},
                 "learning_rate": {"float": [0.01, 0.3], "log": True}, "subsample": {"float": [0.6, 1.0]},
                 "colsample_bytree": {"float": [0.5, 1.0]}, "min_child_samples": {"int": [5, 100]}},
    "mlp": {"alpha": {"float": [1e-5, 1e-1], "log": True}},
    "ridge": {"alpha": {"float": [1e-3, 100.0], "log": True}},
    "lasso": {"alpha": {"float": [1e-4, 10.0], "log": True}},
    "elastic_net": {"alpha": {"float": [1e-4, 10.0], "log": True}, "l1_ratio": {"float": [0.05, 0.95]}},
    "svr": {"C": {"float": [1.0, 1000.0], "log": True}, "epsilon": {"float": [0.01, 10.0], "log": True}},
    "tfidf_logreg": {"C": {"float": [0.1, 100.0], "log": True}},
    "tfidf_svm": {"C": {"float": [0.05, 10.0], "log": True}},
    "tfidf_naive_bayes": {"alpha": {"float": [0.01, 1.0], "log": True}},
    "knn_retrieval": {"k": {"int": [3, 30]}},
}

# Rang de complexité : à égalité statistique, le plus petit rang gagne.
COMPLEXITY = {
    "dummy": 0, "logistic_regression": 1, "linear_regression": 1, "ridge": 1, "lasso": 1,
    "elastic_net": 1, "naive_bayes": 1, "tfidf_naive_bayes": 1, "tfidf_logreg": 1,
    "decision_tree": 2, "knn": 2, "tfidf_svm": 2, "svm": 3, "svr": 3, "knn_retrieval": 3,
    "random_forest": 4, "extra_trees": 4, "hist_gradient_boosting": 5, "xgboost": 5,
    "lightgbm": 5, "mlp": 6, "torch_mlp": 7,
}


def budget(cfg: UseCaseConfig) -> tuple[int, float]:
    trials = int(os.environ.get("FINML_TUNING_TRIALS", cfg.tuning.n_trials))
    return trials, cfg.tuning.timeout_seconds


def _suggest(trial, name: str, space) -> object:
    if space.choice is not None:
        return trial.suggest_categorical(name, space.choice)
    if space.int_range is not None:
        return trial.suggest_int(name, *space.int_range, log=space.log)
    return trial.suggest_float(name, *space.float_range, log=space.log)


def tune(cfg: UseCaseConfig, cand: Candidate, X, y, cross_validate, fold_score) -> tuple[dict, list[dict]]:
    """Cherche le meilleur réglage. Renvoie (params trouvés, essais) ; à appeler dans le run du candidat.

    fold_score(cand, X, y, train_idx, valid_idx) -> score d'un pli ; cross_validate fournit les plis.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = list(cross_validate(X, y))
    maximize = cfg.metric not in LOWER_IS_BETTER
    study = optuna.create_study(
        direction="maximize" if maximize else "minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg.data.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )

    def objective(trial):
        params = {**cand.params, **{k: _suggest(trial, k, s) for k, s in cand.search.items()}}
        trial_cand = cand.model_copy(update={"params": params})
        scores = []
        with mlflow.start_run(run_name=f"{cand.name}-essai-{trial.number}", nested=True):
            mlflow.log_params({k: params[k] for k in cand.search})
            for step, (tr, va) in enumerate(folds):
                scores.append(fold_score(trial_cand, X, y, tr, va))
                trial.report(float(np.mean(scores)), step)
                if trial.should_prune():
                    mlflow.set_tag("etat", "elague")
                    mlflow.log_metric("cv_partielle", float(np.mean(scores)))
                    raise optuna.TrialPruned()
            mlflow.set_tag("etat", "complet")
            mlflow.log_metric(f"cv_{cfg.metric}_mean", float(np.mean(scores)))
        return float(np.mean(scores))

    n_trials, timeout = budget(cfg)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"essais_{cand.name}.csv"
        study.trials_dataframe().to_csv(path, index=False)
        mlflow.log_artifact(str(path))
    trials = [{"numero": t.number, "etat": t.state.name, "valeur": t.value, **t.params} for t in study.trials]
    if not complete:
        return {}, trials
    return {**cand.params, **study.best_params}, trials


def pick_champion(cfg: UseCaseConfig, board: list[dict]) -> tuple[str, str | None]:
    """Champion = meilleur en CV, sauf si un candidat plus simple est à égalité avec lui.

    Renvoie (champion, explication si la règle de simplicité a joué).
    """
    best = board[0]
    sel = cfg.selection
    if not sel.prefer_simpler:
        return best["candidat"], None
    tied = [r for r in board
            if r["candidat"] == best["candidat"]
            or ((r.get("p_value_vs_champion") or 0) > sel.alpha
                and abs(r["cv_moyenne"] - best["cv_moyenne"]) <= sel.tolerance)]
    tied = [r for r in tied if r["type"] != "dummy"]
    chosen = min(tied, key=lambda r: (COMPLEXITY.get(r["type"], 9), board.index(r)))
    if chosen["candidat"] == best["candidat"]:
        return best["candidat"], None
    return chosen["candidat"], (f"{chosen['candidat']} retenu au lieu de {best['candidat']} : égalité statistique "
                                f"(écart CV {best['cv_moyenne'] - chosen['cv_moyenne']:+.4f}), modèle plus simple")

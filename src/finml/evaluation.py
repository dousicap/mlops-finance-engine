"""Évaluation complète du champion sur le jeu de test, au seuil réellement servi.

Classification binaire (seuil métier) :
- matrice de confusion en effectifs (VP, FP, FN, VN) ;
- précision, rappel (= sensibilité), spécificité, F1, exactitude, exactitude équilibrée ;
- coefficient de corrélation de Matthews (MCC) : élevé seulement si les deux classes sont bien
  prédites, donc fiable sur données déséquilibrées (Chicco et Jurman, 2020) ;
- indépendantes du seuil : ROC AUC, PR AUC (average precision), Gini, statistique KS
  (standards du scoring de crédit), log loss et score de Brier (qualité des probabilités).
Multi-classes : exactitude, exactitude équilibrée, F1 et précision / rappel macro et pondérés,
MCC, kappa de Cohen, log loss, top 3, rapport par classe.
Régression : RMSE, MAE, R2, erreur absolue médiane, erreur maximale, MAPE (sans cible nulle).

Fuite de la cible : chaque variable seule, puis toutes ensemble, sont confrontées à la cible
(arbre de profondeur 4 par variable, 8 pour l'ensemble, validation croisée). Un score presque parfait avec un modèle aussi simple
signale presque toujours une donnée connue après coup (ex. les modes de panne en maintenance,
dont la combinaison reconstitue la panne). C'est une alerte à vérifier, pas un blocage.

Robustesse : intervalle de confiance à 95 % de la métrique principale par bootstrap du jeu de test.
Comparaison des candidats : test t corrigé de Nadeau et Bengio (2003) sur les plis de validation
croisée. Les plis partagent des données d'entraînement : un test t naïf déclarerait trop souvent
une différence qui n'existe pas.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn import metrics as m

from finml.config import UseCaseConfig
from finml.models import class_scores, is_binary

N_BOOTSTRAP = 500


def _binary(y, proba, threshold) -> tuple[dict, dict]:
    y = np.asarray(y).astype(int)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = m.confusion_matrix(y, pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = m.roc_curve(y, proba)
    auc = m.roc_auc_score(y, proba)
    out = {
        "seuil": float(threshold),
        "vrais_positifs": int(tp), "faux_positifs": int(fp), "faux_negatifs": int(fn), "vrais_negatifs": int(tn),
        "precision": float(m.precision_score(y, pred, zero_division=0)),
        "rappel": float(m.recall_score(y, pred, zero_division=0)),
        "specificite": float(tn / max(tn + fp, 1)),
        "f1": float(m.f1_score(y, pred, zero_division=0)),
        "exactitude": float(m.accuracy_score(y, pred)),
        "exactitude_equilibree": float(m.balanced_accuracy_score(y, pred)),
        "mcc": float(m.matthews_corrcoef(y, pred)),
        "roc_auc": float(auc),
        "pr_auc": float(m.average_precision_score(y, proba)),
        "gini": float(2 * auc - 1),
        "ks": float(np.max(tpr - fpr)),
        "log_loss": float(m.log_loss(y, np.clip(proba, 1e-15, 1 - 1e-15), labels=[0, 1])),
        "brier": float(m.brier_score_loss(y, proba)),
    }
    return out, {}


def _multiclass(y, classes, scores) -> tuple[dict, dict]:
    y = np.asarray(y).astype(str)
    classes = np.asarray(classes).astype(str)
    pred = classes[scores.argmax(axis=1)]
    top3 = classes[np.argsort(-scores, axis=1)[:, :3]]
    probs = np.clip(scores / scores.sum(axis=1, keepdims=True).clip(min=1e-12), 1e-15, 1)
    out = {
        "exactitude": float(m.accuracy_score(y, pred)),
        "exactitude_equilibree": float(m.balanced_accuracy_score(y, pred)),
        "f1_macro": float(m.f1_score(y, pred, average="macro", zero_division=0)),
        "f1_pondere": float(m.f1_score(y, pred, average="weighted", zero_division=0)),
        "precision_macro": float(m.precision_score(y, pred, average="macro", zero_division=0)),
        "rappel_macro": float(m.recall_score(y, pred, average="macro", zero_division=0)),
        "mcc": float(m.matthews_corrcoef(y, pred)),
        "kappa": float(m.cohen_kappa_score(y, pred)),
        "log_loss": float(m.log_loss(y, probs, labels=classes)),
        "top3": float(np.mean([y[i] in top3[i] for i in range(len(y))])),
    }
    per_class = m.classification_report(y, pred, labels=classes, output_dict=True, zero_division=0)
    return out, {"par_classe": per_class}


def _regression(y, pred) -> tuple[dict, dict]:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    out = {
        "rmse": float(m.root_mean_squared_error(y, pred)),
        "mae": float(m.mean_absolute_error(y, pred)),
        "r2": float(m.r2_score(y, pred)),
        "erreur_absolue_mediane": float(m.median_absolute_error(y, pred)),
        "erreur_max": float(m.max_error(y, pred)),
    }
    if not np.any(y == 0):
        out["mape"] = float(m.mean_absolute_percentage_error(y, pred))
    return out, {}


def _main_metric(cfg: UseCaseConfig, y, kind: str, payload) -> float:
    """Recalcule la métrique de sélection sur un échantillon (pour le bootstrap)."""
    if kind == "regression":
        return {"rmse": lambda: m.root_mean_squared_error(y, payload),
                "mae": lambda: m.mean_absolute_error(y, payload),
                "r2": lambda: m.r2_score(y, payload)}[cfg.metric]()
    if kind == "binary":
        proba, threshold = payload
        pred = (proba >= threshold).astype(int)
        return {"roc_auc": lambda: m.roc_auc_score(y, proba),
                "average_precision": lambda: m.average_precision_score(y, proba),
                "f1": lambda: m.f1_score(y, pred, zero_division=0),
                "accuracy": lambda: m.accuracy_score(y, pred)}[cfg.metric]()
    classes, scores = payload
    pred = classes[scores.argmax(axis=1)]
    if cfg.metric == "top3_accuracy":
        top3 = classes[np.argsort(-scores, axis=1)[:, :3]]
        return float(np.mean([y[i] in top3[i] for i in range(len(y))]))
    if cfg.metric == "f1":
        return m.f1_score(y, pred, average="macro", zero_division=0)
    if cfg.metric == "roc_auc":
        return m.roc_auc_score(y, scores, multi_class="ovr", average="weighted", labels=classes)
    return m.accuracy_score(y, pred)


def _bootstrap(cfg: UseCaseConfig, y, kind: str, payload, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    values = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        if kind != "regression" and len(np.unique(y[idx])) < 2:
            continue
        if kind == "regression":
            sub = payload[idx]
        elif kind == "binary":
            sub = (payload[0][idx], payload[1])
        else:
            sub = (payload[0], payload[1][idx])
        try:
            values.append(float(_main_metric(cfg, y[idx], kind, sub)))
        except ValueError:  # une classe absente de l'échantillon : tirage ignoré
            continue
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def evaluate_champion(cfg: UseCaseConfig, est, threshold: float, X_test, y_test) -> tuple[dict, dict]:
    """Renvoie (métriques eval_* à tracer, détail JSON)."""
    if cfg.is_regression:
        payload = np.asarray(est.predict(X_test), float)
        kind, (out, detail) = "regression", _regression(y_test, payload)
    elif cfg.is_tabular and is_binary(est):
        proba = est.predict_proba(X_test)[:, 1]
        payload = (proba, threshold)
        kind, (out, detail) = "binary", _binary(y_test, proba, threshold)
    else:
        classes, scores = class_scores(est, cfg, X_test)
        payload = (np.asarray(classes), np.asarray(scores))
        kind, (out, detail) = "multiclass", _multiclass(y_test, classes, scores)
    y = np.asarray(y_test).astype(str) if kind == "multiclass" else np.asarray(y_test)
    if kind == "multiclass":
        payload = (payload[0].astype(str), payload[1])
    low, high = _bootstrap(cfg, y, kind, payload, cfg.data.seed)
    out[f"{cfg.metric}_ic95_bas"], out[f"{cfg.metric}_ic95_haut"] = low, high
    detail["metriques"] = out
    return {f"eval_{k}": v for k, v in out.items()}, detail


def corrected_t_test(a: np.ndarray, b: np.ndarray, test_size_ratio: float) -> float:
    """p-value bilatérale du test t corrigé de Nadeau et Bengio sur des scores de plis appariés.

    test_size_ratio = n_test / n_train d'un pli, soit 1 / (k - 1) en validation croisée à k plis.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    k = len(d)
    var = d.var(ddof=1)
    if k < 2 or var == 0:
        return 1.0
    t = d.mean() / np.sqrt((1 / k + test_size_ratio) * var)
    return float(2 * stats.t.sf(abs(t), df=k - 1))


def compare_to_champion(board: list[dict], fold_scores: dict, cfg: UseCaseConfig,
                        champion: str | None = None) -> None:
    """Ajoute à chaque ligne du classement la p-value de l'écart avec le champion
    (par défaut le meilleur en validation croisée)."""
    champion = champion or board[0]["candidat"]
    ratio = 1 / (cfg.cv_folds - 1)
    for row in board:
        row["p_value_vs_champion"] = (
            None if row["candidat"] == champion
            else round(corrected_t_test(fold_scores[champion], fold_scores[row["candidat"]], ratio), 4)
        )


LEAK_THRESHOLD = 0.98


def leakage_check(cfg: UseCaseConfig, X, y) -> dict[str, float]:
    """Variables qui, seules, prédisent la cible presque parfaitement (AUC, exactitude ou R2 >= 0,98)."""
    import pandas as pd
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    if cfg.is_regression:
        model, scoring = DecisionTreeRegressor(max_depth=4, random_state=0), "r2"
    else:
        model = DecisionTreeClassifier(max_depth=4, random_state=0)
        scoring = "roc_auc" if pd.Series(y).nunique() == 2 else "accuracy"
    # Découpage mélangé : un fichier trié (par date, par machine) fausserait les plis.
    folds = (KFold if cfg.is_regression else StratifiedKFold)(3, shuffle=True, random_state=cfg.data.seed)
    encoded = pd.DataFrame({
        col: pd.factorize(X[col])[0] if col in cfg.features.categorical else X[col] for col in X.columns
    }).fillna(-999)
    suspects = {}
    for col in encoded.columns:
        score = cross_val_score(model, encoded[[col]], y, cv=folds, scoring=scoring).mean()
        if score >= LEAK_THRESHOLD:
            suspects[col] = round(float(score), 4)
    # Fuite par combinaison : plusieurs colonnes qui, ensemble, reconstituent la cible
    # (ex. les modes de panne, dont l'union est la panne). Un simple arbre ne devrait pas
    # atteindre un score presque parfait sur un problème réel.
    deep = model.set_params(max_depth=8)  # assez profond pour combiner plusieurs colonnes
    score = cross_val_score(deep, encoded, y, cv=folds, scoring=scoring).mean()
    if score >= LEAK_THRESHOLD and not suspects:
        tree = deep.fit(encoded, y)
        top = encoded.columns[tree.feature_importances_.argsort()[::-1][:3]]
        suspects["combinaison : " + " + ".join(top)] = round(float(score), 4)
    return suspects

"""`finml init` : crée un cas d'usage à partir d'un CSV et du nom de la colonne cible.

Détection, toujours écrite en commentaire dans la configuration générée pour être relue :
- tâche : régression si la cible est numérique avec plus de 20 valeurs distinctes ou des
  décimales, classification sinon (binaire ou multi-classes selon le nombre de classes) ;
  classification de texte si la seule variable exploitable est un texte libre ;
- colonnes : numériques, catégorielles (50 modalités au plus), texte libre (longueur moyenne
  de 20 caractères ou plus), exclues (identifiant unique par ligne, constante, forte cardinalité) ;
- porte qualité : battre une référence naïve (R2 > 0, AUC > 0,55, exactitude > classe majoritaire) ;
- candidats : tout l'arsenal compatible avec la tâche, réglages par défaut raisonnables.

Le CSV est copié dans le dossier du cas d'usage : la configuration et ses données sont versionnées
ensemble, l'entraînement est reproductible.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pandas as pd
import yaml

from finml.config import UseCaseConfig, project_root, usecases_root

MAX_CATEGORIES = 50
MIN_TEXT_LENGTH = 20
REGRESSION_MIN_UNIQUE = 20

CANDIDATES = {
    "tabular_regression": [
        ("regression_lineaire", "linear_regression", {}),
        ("ridge", "ridge", {"alpha": 1.0}),
        ("lasso", "lasso", {"alpha": 0.1}),
        ("elastic_net", "elastic_net", {"alpha": 0.1, "l1_ratio": 0.5}),
        ("svr", "svr", {"C": 100.0}),
        ("k_plus_proches_voisins", "knn", {"n_neighbors": 15}),
        ("arbre_de_decision", "decision_tree", {"max_depth": 6, "min_samples_leaf": 5}),
        ("random_forest", "random_forest", {"n_estimators": 300, "min_samples_leaf": 3}),
        ("extra_trees", "extra_trees", {"n_estimators": 300, "min_samples_leaf": 3}),
        ("gradient_boosting_histogramme", "hist_gradient_boosting", {"max_iter": 300, "learning_rate": 0.05}),
        ("xgboost", "xgboost", {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05}),
        ("lightgbm", "lightgbm", {"n_estimators": 300, "num_leaves": 15, "learning_rate": 0.05}),
        ("reseau_neurones_sklearn", "mlp", {"hidden_layer_sizes": [64, 32], "alpha": 0.001}),
    ],
    "tabular_classification": [
        ("regression_logistique", "logistic_regression", {"C": 1.0}),
        ("naive_bayes", "naive_bayes", {}),
        ("k_plus_proches_voisins", "knn", {"n_neighbors": 15}),
        ("svm_lineaire", "svm", {"C": 1.0}),
        ("arbre_de_decision", "decision_tree", {"max_depth": 6, "min_samples_leaf": 5}),
        ("random_forest", "random_forest", {"n_estimators": 300, "min_samples_leaf": 3}),
        ("extra_trees", "extra_trees", {"n_estimators": 300, "min_samples_leaf": 3}),
        ("gradient_boosting_histogramme", "hist_gradient_boosting", {"max_iter": 300, "learning_rate": 0.05}),
        ("xgboost", "xgboost", {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05}),
        ("lightgbm", "lightgbm", {"n_estimators": 300, "num_leaves": 15, "learning_rate": 0.05}),
        ("reseau_neurones_sklearn", "mlp", {"hidden_layer_sizes": [64, 32], "alpha": 0.001}),
        ("reseau_neurones_pytorch", "torch_mlp", {"hidden": [64, 32], "dropout": 0.2, "epochs": 60}),
    ],
    "text_classification": [
        ("tfidf_regression_logistique", "tfidf_logreg", {"C": 10.0}),
        ("tfidf_svm_lineaire", "tfidf_svm", {"C": 1.0}),
        ("tfidf_naive_bayes", "tfidf_naive_bayes", {"alpha": 0.1}),
        ("recherche_vectorielle_k5", "knn_retrieval", {"k": 5}),
    ],
}


def _py(values) -> list:
    """Valeurs numpy vers types Python simples, triées, pour un YAML lisible."""
    out = [v.item() if hasattr(v, "item") else v for v in values]
    return sorted(out, key=str)


def infer(df: pd.DataFrame, target: str) -> dict:
    """Analyse un jeu de données. Renvoie la tâche, les colonnes par type et les exclusions."""
    if target not in df.columns:
        raise ValueError(f"Colonne cible absente : {target}. Colonnes : {list(df.columns)}")
    y = df[target]
    if y.isna().any():
        raise ValueError(f"La cible {target} a {int(y.isna().sum())} valeur(s) manquante(s)")
    numeric_target = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)
    has_decimals = numeric_target and bool((y.astype(float) % 1 != 0).any())
    regression = numeric_target and (y.nunique() > REGRESSION_MIN_UNIQUE or has_decimals)

    numeric, categorical, text, excluded = [], [], [], {}
    n = len(df)
    for col in df.columns.drop(target):
        s = df[col]
        if s.nunique(dropna=True) <= 1:
            excluded[col] = "constante"
        elif pd.api.types.is_numeric_dtype(s):
            if pd.api.types.is_integer_dtype(s) and s.nunique() == n:
                excluded[col] = "identifiant (valeur unique par ligne)"
            else:
                numeric.append(col)
        elif s.nunique() == n and s.astype(str).str.len().mean() < MIN_TEXT_LENGTH:
            excluded[col] = "identifiant (valeur unique par ligne)"
        elif s.nunique() <= MAX_CATEGORIES:
            categorical.append(col)
        elif s.astype(str).str.len().mean() >= MIN_TEXT_LENGTH:
            text.append(col)
        else:
            excluded[col] = f"plus de {MAX_CATEGORIES} modalités, texte trop court"

    if regression:
        task = "tabular_regression"
    elif text and not numeric and not categorical:
        task = "text_classification"
    else:
        task = "tabular_classification"
    if task != "text_classification":
        for col in text:
            excluded[col] = "texte libre : non utilisé dans un modèle tabulaire"
        text = []
    return {"task": task, "numeric": numeric, "categorical": categorical, "text": text[:1],
            "excluded": excluded, "n_classes": None if regression else int(y.nunique())}


def build_config(df: pd.DataFrame, name: str, target: str, data_path: str) -> tuple[dict, dict]:
    info = infer(df, target)
    task, y = info["task"], df[target]
    if task == "tabular_regression":
        metric, gate = "r2", {"min_score": 0.0}
    elif info["n_classes"] == 2 and task == "tabular_classification":
        metric, gate = "roc_auc", {"min_score": 0.55}
    else:
        majority = float(y.value_counts(normalize=True).iloc[0])
        metric, gate = "accuracy", {"min_score": round(majority, 2)}

    schema = {}
    for col in info["numeric"]:
        if df[col].isna().any():
            schema[col] = {"nullable": True}
    for col in info["categorical"]:
        rule = {"allowed": _py(df[col].dropna().unique())}
        if df[col].isna().any():
            rule["nullable"] = True
        schema[col] = rule
    if task == "text_classification":  # la cible texte est lue en chaînes de caractères
        schema[target] = {"allowed": sorted(y.astype(str).unique().tolist())}
    elif task != "tabular_regression":
        schema[target] = {"allowed": _py(y.unique())}

    folds = 5
    if task != "tabular_regression":
        folds = max(2, min(5, int(y.value_counts().min())))
    features = {"numeric": info["numeric"], "categorical": info["categorical"]}
    if info["text"]:
        features = {"text": info["text"][0]}
    cfg = {
        "name": name,
        "description": f"Prédiction de {target} ({task}), cas créé par finml init.",
        "task": task,
        "registered_model": "finml-" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-"),
        "target": target,
        "features": features,
        "schema": schema,
        "data": {"path": data_path, "n_rows": len(df), "seed": 42},
        "metric": metric,
        **gate,
        "cv_folds": folds,
        "candidates": [{"name": n, "type": t, "params": p} for n, t, p in CANDIDATES[task]],
    }
    UseCaseConfig.model_validate(cfg)  # la configuration générée doit être valide
    return cfg, info


def init_usecase(name: str, csv: Path, target: str, force: bool = False) -> tuple[Path, dict]:
    folder = usecases_root() / name
    if (folder / "config.yaml").exists() and not force:
        raise FileExistsError(f"{folder}/config.yaml existe déjà (--force pour remplacer)")
    df = pd.read_csv(csv)
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / "donnees.csv"
    if Path(csv).resolve() != dest.resolve():
        shutil.copyfile(csv, dest)
    cfg, info = build_config(df, name, target, str(dest.relative_to(project_root())))
    header = [
        f"# Généré par `finml init {name} --csv {Path(csv).name} --cible {target}`.",
        f"# {len(df)} lignes. Tâche détectée : {info['task']}"
        + (f", {info['n_classes']} classes." if info["n_classes"] else "."),
        "# Porte qualité : battre une référence naïve. Relire avant la mise en production.",
    ]
    header += [f"# Colonne exclue : {c} ({why})" for c, why in info["excluded"].items()]
    body = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False, width=100)
    (folder / "config.yaml").write_text("\n".join(header) + "\n\n" + body, encoding="utf-8")
    return folder / "config.yaml", info

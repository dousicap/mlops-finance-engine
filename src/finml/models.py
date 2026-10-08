"""Fabrique de modèles candidats, sérialisation sûre (skops) et post-traitement commun.

Aucun pickle sur le chemin de service : les estimateurs sont sérialisés avec skops et
ne sont rechargés que si tous leurs types figurent dans TRUSTED_TYPES. L'enveloppe MLflow
est chargée depuis son code source (« models from code », MLflow 3), pas désérialisée.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import skops.io as sio
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from finml.config import Candidate, UseCaseConfig
from finml.vector import VectorRetrievalClassifier


def build_estimator(cfg: UseCaseConfig, cand: Candidate):
    p = dict(cand.params)
    if cfg.task == "tabular_classification":
        pre = ColumnTransformer(
            [
                ("num", StandardScaler(), cfg.features.numeric),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cfg.features.categorical),
            ]
        )
        if cand.type == "logistic_regression":
            est = LogisticRegression(max_iter=2000, **p)
        elif cand.type == "random_forest":
            est = RandomForestClassifier(n_jobs=-1, random_state=cfg.data.seed, **p)
        elif cand.type == "xgboost":
            est = XGBClassifier(eval_metric="logloss", random_state=cfg.data.seed, n_jobs=-1, **p)
        else:
            raise ValueError(f"{cand.type} n'est pas un modèle tabulaire")
        return Pipeline([("pre", pre), ("model", est)])

    if cand.type == "tfidf_logreg":
        ngram = tuple(p.pop("ngram_range", (2, 5)))
        return Pipeline(
            [
                ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, strip_accents="unicode",
                                          sublinear_tf=True)),
                ("model", LogisticRegression(max_iter=3000, **p)),
            ]
        )
    if cand.type == "knn_retrieval":
        return VectorRetrievalClassifier(**p)
    raise ValueError(f"{cand.type} n'est pas un modèle texte")


def text_class_scores(estimator, texts) -> tuple[np.ndarray, np.ndarray]:
    """Scores par classe pour un modèle texte, quel que soit son type."""
    if isinstance(estimator, VectorRetrievalClassifier):
        return estimator.classes_, estimator.class_scores(texts)
    return estimator.classes_, estimator.predict_proba(texts)


# Types hors bibliothèque standard scikit-learn que skops doit accepter. Toute autre classe
# rencontrée au chargement est refusée : un fichier de modèle altéré ne peut pas exécuter de code.
TRUSTED_TYPES = frozenset(
    {
        "sklearn.compose._column_transformer._RemainderColsList",
        "sklearn.tree._tree.Tree",
        "xgboost.core.Booster",
        "xgboost.sklearn.XGBClassifier",
        "finml.vector.TfidfEmbedder",
        "finml.vector.SentenceTransformerEmbedder",
        "finml.vector.VectorRetrievalClassifier",
    }
)


class UntrustedModelError(RuntimeError):
    """Le fichier de modèle contient un type non autorisé."""


def dump_estimator(estimator, path: str | Path) -> None:
    sio.dump(estimator, path)


def load_estimator(path: str | Path):
    unknown = sio.get_untrusted_types(file=path)
    refused = sorted(set(unknown) - TRUSTED_TYPES)
    if refused:
        raise UntrustedModelError(f"Types non autorisés dans {path} : {refused}")
    return sio.load(path, trusted=unknown)


def predict_frame(estimator, cfg: UseCaseConfig, model_input: pd.DataFrame) -> pd.DataFrame:
    """Post-traitement commun : même format de sortie pour tous les cas d'usage d'un type."""
    if cfg.task == "tabular_classification":
        X = model_input[cfg.features.all]
        proba = estimator.predict_proba(X)[:, 1]
        return pd.DataFrame(
            {
                "probabilite": np.round(proba, 4),
                "prediction": (proba >= cfg.decision_threshold).astype(int),
                "seuil": cfg.decision_threshold,
            }
        )

    labels = cfg.labels
    texts = model_input[cfg.features.text].astype(str).tolist()
    classes, scores = text_class_scores(estimator, texts)
    top = np.argsort(-scores, axis=1)[:, :3]
    explanations = estimator.explain(texts) if isinstance(estimator, VectorRetrievalClassifier) else None
    rows = []
    for i in range(len(texts)):
        top3 = [
            {
                "compte": str(classes[j]),
                "intitule": labels.get(str(classes[j]), ""),
                "score": round(float(scores[i, j]), 4),
            }
            for j in top[i]
        ]
        rows.append(
            {
                "compte": top3[0]["compte"],
                "intitule": top3[0]["intitule"],
                "confiance": top3[0]["score"],
                "a_revoir": bool(top3[0]["score"] < cfg.decision_threshold),
                "top3": json.dumps(top3, ensure_ascii=False),
                "exemples_proches": json.dumps(explanations[i] if explanations else [], ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)

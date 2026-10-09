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
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from finml.config import Candidate, UseCaseConfig
from finml.vector import VectorRetrievalClassifier


class EncodedLabelClassifier(ClassifierMixin, BaseEstimator):
    """Encode les classes en 0..K-1 avant l'entraînement (exigé par XGBoost), les restitue en sortie."""

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        self.classes_, codes = np.unique(np.asarray(y), return_inverse=True)
        self.estimator_ = clone(self.estimator).fit(X, codes)
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def _regression_model(cand: Candidate, seed: int):
    p = dict(cand.params)
    t = cand.type
    if t == "linear_regression":
        return LinearRegression(**p)
    if t == "ridge":
        return Ridge(random_state=seed, **p)
    if t == "lasso":
        return Lasso(random_state=seed, max_iter=10000, **p)
    if t == "elastic_net":
        return ElasticNet(random_state=seed, max_iter=10000, **p)
    if t == "svr":
        return SVR(**p)
    if t == "knn":
        return KNeighborsRegressor(**p)
    if t == "decision_tree":
        return DecisionTreeRegressor(random_state=seed, **p)
    if t == "random_forest":
        return RandomForestRegressor(n_jobs=-1, random_state=seed, **p)
    if t == "extra_trees":
        return ExtraTreesRegressor(n_jobs=-1, random_state=seed, **p)
    if t == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(random_state=seed, **p)
    if t == "xgboost":
        return XGBRegressor(random_state=seed, n_jobs=-1, **p)
    if t == "lightgbm":
        return LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1, **p)
    if t == "mlp":
        return MLPRegressor(random_state=seed, early_stopping=True, max_iter=1000, **p)
    raise ValueError(f"{t} n'est pas un modèle de régression")


def _tabular_model(cand: Candidate, seed: int):
    p = dict(cand.params)
    t = cand.type
    if t == "logistic_regression":
        return LogisticRegression(max_iter=2000, **p)
    if t == "naive_bayes":
        return GaussianNB(**p)
    if t == "knn":
        return KNeighborsClassifier(**p)
    if t == "svm":  # SVM linéaire, probabilités calibrées par validation croisée
        return CalibratedClassifierCV(LinearSVC(C=p.pop("C", 1.0), random_state=seed), cv=3, **p)
    if t == "decision_tree":
        return DecisionTreeClassifier(random_state=seed, **p)
    if t == "random_forest":
        return RandomForestClassifier(n_jobs=-1, random_state=seed, **p)
    if t == "extra_trees":
        return ExtraTreesClassifier(n_jobs=-1, random_state=seed, **p)
    if t == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=seed, **p)
    if t == "xgboost":
        return EncodedLabelClassifier(XGBClassifier(random_state=seed, n_jobs=-1, **p))
    if t == "lightgbm":
        return LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1, **p)
    if t == "mlp":
        return MLPClassifier(random_state=seed, early_stopping=True, max_iter=500, **p)
    if t == "torch_mlp":
        from finml.torch_model import TorchMLPClassifier

        if "hidden" in p:
            p["hidden"] = tuple(p["hidden"])
        return TorchMLPClassifier(seed=seed, **p)
    raise ValueError(f"{t} n'est pas un modèle tabulaire")


# Modèles qui exigent une matrice dense en entrée.
DENSE_INPUT = {"naive_bayes", "hist_gradient_boosting", "torch_mlp"}


BASELINE = "reference_naive"


def build_estimator(cfg: UseCaseConfig, cand: Candidate):
    if cand.type == "dummy":  # référence naïve : moyenne ou fréquence des classes, sans regarder X
        return DummyRegressor(strategy="mean") if cfg.is_regression else DummyClassifier(strategy="prior")
    if cfg.is_tabular:
        dense = cand.type in DENSE_INPUT
        # Valeurs manquantes : médiane (numérique), modalité la plus fréquente (catégorielle).
        # Sans valeur manquante, l'imputation ne change rien.
        num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        cat = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)),
        ])
        pre = ColumnTransformer(
            [("num", num, cfg.features.numeric), ("cat", cat, cfg.features.categorical)],
            sparse_threshold=0.0 if dense else 0.3,
        )
        model = (_regression_model if cfg.is_regression else _tabular_model)(cand, cfg.data.seed)
        return Pipeline([("pre", pre), ("model", model)])

    p = dict(cand.params)
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(p.pop("ngram_range", (2, 5))),
                            strip_accents="unicode", sublinear_tf=True)
    if cand.type == "tfidf_logreg":
        return Pipeline([("tfidf", tfidf), ("model", LogisticRegression(max_iter=3000, **p))])
    if cand.type == "tfidf_svm":
        svm = CalibratedClassifierCV(LinearSVC(C=p.pop("C", 1.0), random_state=cfg.data.seed), cv=3)
        return Pipeline([("tfidf", tfidf), ("model", svm)])
    if cand.type == "tfidf_naive_bayes":
        return Pipeline([("tfidf", tfidf), ("model", ComplementNB(**p))])
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
        "xgboost.sklearn.XGBRegressor",
        "lightgbm.basic.Booster",
        "lightgbm.sklearn.LGBMClassifier",
        "lightgbm.sklearn.LGBMRegressor",
        "collections.OrderedDict",
        "sklearn.calibration._CalibratedClassifier",
        "sklearn.calibration._SigmoidCalibration",
        "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
        "sklearn.neural_network._stochastic_optimizers.AdamOptimizer",
        # KNN : l'algorithme « auto » construit un arbre selon la dimension des données
        "sklearn.neighbors._kd_tree.KDTree",
        "sklearn.neighbors._ball_tree.BallTree",
        "sklearn.metrics._dist_metrics.EuclideanDistance64",
        "finml.torch_model.TorchMLPClassifier",
        "finml.models.EncodedLabelClassifier",
        # SimpleImputer mémorise le type des données vues (numpy.dtype) : objet de description, sans code
        "numpy.dtype",
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


def is_binary(estimator) -> bool:
    return len(getattr(estimator, "classes_", [])) == 2


def class_scores(estimator, cfg: UseCaseConfig, X) -> tuple[np.ndarray, np.ndarray]:
    """Scores par classe, pour un modèle texte ou tabulaire multi-classes."""
    if cfg.task == "text_classification":
        return text_class_scores(estimator, X)
    return estimator.classes_, estimator.predict_proba(X)


def predict_frame(estimator, cfg: UseCaseConfig, model_input: pd.DataFrame) -> pd.DataFrame:
    """Post-traitement commun : même format de sortie pour tous les cas d'usage d'un type."""
    if cfg.is_regression:
        pred = estimator.predict(model_input[cfg.features.all])
        return pd.DataFrame({"valeur": np.round(pred.astype(float), 4)})
    if cfg.is_tabular and is_binary(estimator):
        X = model_input[cfg.features.all]
        proba = estimator.predict_proba(X)[:, 1]
        pred = (proba >= cfg.decision_threshold).astype(int)
        frame = pd.DataFrame({"probabilite": np.round(proba, 4), "prediction": pred, "seuil": cfg.decision_threshold})
        if cfg.labels:  # cible d'origine non 0/1 (ex. oui / non) : libellé de la classe prédite
            frame["classe"] = [cfg.labels[str(v)] for v in pred]
        return frame

    labels, out = cfg.labels, cfg.output
    if cfg.is_tabular:
        X, explanations = model_input[cfg.features.all], None
    else:
        X = model_input[cfg.features.text].astype(str).tolist()
        explanations = estimator.explain(X) if isinstance(estimator, VectorRetrievalClassifier) else None
    classes, scores = class_scores(estimator, cfg, X)
    top = np.argsort(-scores, axis=1)[:, :3]
    rows = []
    for i in range(len(scores)):
        top3 = [
            {
                out.class_field: str(classes[j]),
                out.class_name_field: labels.get(str(classes[j]), ""),
                "score": round(float(scores[i, j]), 4),
            }
            for j in top[i]
        ]
        row = {
            out.class_field: top3[0][out.class_field],
            out.class_name_field: top3[0][out.class_name_field],
            "confiance": top3[0]["score"],
            "a_revoir": bool(top3[0]["score"] < cfg.decision_threshold),
            "top3": json.dumps(top3, ensure_ascii=False),
        }
        if not cfg.is_tabular:
            # Écritures passées les plus proches, nommées comme les champs du cas d'usage.
            near = [{cfg.features.text: e["texte"], out.class_field: e["classe"], "similarite": e["similarite"]}
                    for e in (explanations[i] if explanations else [])]
            row["exemples_proches"] = json.dumps(near, ensure_ascii=False)
        rows.append(row)
    return pd.DataFrame(rows)

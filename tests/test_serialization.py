"""Chaque candidat déclaré doit survivre à l'aller-retour skops : sinon le champion serait
refusé au chargement en production (liste blanche de types)."""

import importlib.util

import numpy as np
import pytest

from finml import data
from finml.config import load_config
from finml.models import build_estimator, dump_estimator, load_estimator, text_class_scores


@pytest.mark.parametrize("usecase", ["retard_paiement", "imputation_comptable"])
def test_every_candidate_roundtrips_through_skops(project, tmp_path, usecase):
    cfg = load_config(usecase)
    data.generate(cfg)
    df = data.load(cfg).head(600)
    if cfg.task == "tabular_classification":
        X = df[cfg.features.all].astype({c: "float64" for c in cfg.features.numeric})
    else:
        X = df[cfg.features.text].astype(str)
    y = df[cfg.target]
    for cand in cfg.candidates:
        if cand.type == "torch_mlp" and importlib.util.find_spec("torch") is None:
            continue
        est = build_estimator(cfg, cand).fit(X, y)
        path = tmp_path / f"{cand.name}.skops"
        dump_estimator(est, path)
        loaded = load_estimator(path)  # lève UntrustedModelError si un type manque
        if cfg.task == "tabular_classification":
            np.testing.assert_allclose(loaded.predict_proba(X.head(20)), est.predict_proba(X.head(20)), rtol=1e-5)
        else:
            assert (text_class_scores(loaded, X.head(20))[1].argmax(1)
                    == text_class_scores(est, X.head(20))[1].argmax(1)).all()

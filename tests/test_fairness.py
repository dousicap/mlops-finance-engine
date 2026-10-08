"""Audit d'équité : écarts mesurés, petits groupes exclus, colonnes sensibles hors du modèle."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from finml.config import FairnessConfig, UseCaseConfig, load_config
from finml.fairness import fairness_report

CFG = FairnessConfig(sensitive=["groupe"])


def _data(n_a, n_b, recall_a, recall_b):
    """Deux groupes, 50 % de positifs, rappel imposé par groupe, aucun faux positif."""
    y, flag, g = [], [], []
    for name, n, rec in (("a", n_a, recall_a), ("b", n_b, recall_b)):
        pos = n // 2
        y += [1] * pos + [0] * (n - pos)
        flag += [1] * int(pos * rec) + [0] * (pos - int(pos * rec)) + [0] * (n - pos)
        g += [name] * n
    return np.array(y), np.array(flag), pd.DataFrame({"groupe": g})


def test_equal_treatment_is_ok():
    metrics, detail = fairness_report(*_data(400, 400, 0.6, 0.6), CFG)
    assert detail["statut"] == "ok" and metrics["equite_groupe_ecart_rappel"] == 0


def test_recall_gap_raises_an_alert():
    metrics, detail = fairness_report(*_data(400, 400, 0.8, 0.4), CFG)
    assert detail["statut"] == "alerte"
    assert metrics["equite_groupe_ecart_rappel"] == pytest.approx(0.4)


def test_small_group_is_described_but_not_compared():
    _, detail = fairness_report(*_data(400, 40, 0.8, 0.1), CFG)
    assert detail["statut"] == "ok"
    assert detail["groupes"]["groupe"]["b"]["compare"] is False


def test_sensitive_column_cannot_be_a_feature(project):
    raw = load_config("defaut_credit").model_dump(by_alias=True)
    raw["features"]["categorical"].append("sexe")
    with pytest.raises(ValidationError):
        UseCaseConfig.model_validate(raw)

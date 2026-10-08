import pandas as pd
import pytest
from pydantic import ValidationError

from finml import data
from finml.config import UseCaseConfig, list_usecases, load_config


def test_usecases_are_discovered(project):
    assert list_usecases() == ["imputation_comptable", "retard_paiement"]


def test_configs_are_valid(project):
    for name in list_usecases():
        cfg = load_config(name)
        assert cfg.candidates
        assert cfg.target not in cfg.features.all


def test_target_cannot_be_a_feature(project):
    raw = load_config("retard_paiement").model_dump(by_alias=True)
    raw["features"]["numeric"].append(raw["target"])
    with pytest.raises(ValidationError):
        UseCaseConfig.model_validate(raw)


@pytest.mark.parametrize("name", ["retard_paiement", "imputation_comptable"])
def test_generation_is_deterministic_and_valid(project, name):
    cfg = load_config(name)
    data.generate(cfg)
    first = data.fingerprint(data.load(cfg))
    data.generate(cfg)
    df = data.load(cfg)
    assert data.fingerprint(df) == first
    assert len(df) == cfg.data.n_rows
    assert data.validate(df, cfg).ok


def test_quality_checks_catch_bad_rows(project):
    cfg = load_config("retard_paiement")
    data.generate(cfg)
    df = data.load(cfg)
    df.loc[0, "mois_emission"] = 13
    df.loc[1, "secteur"] = "inconnu"
    df.loc[2, "montant_ttc"] = None
    report = data.validate(df, cfg)
    assert not report.ok
    assert any("mois_emission" in e for e in report.errors)
    assert any("secteur" in e for e in report.errors)
    assert any("montant_ttc" in e for e in report.errors)


def test_missing_column_is_reported(project):
    cfg = load_config("retard_paiement")
    report = data.validate(pd.DataFrame({"montant_ttc": [100.0]}), cfg, with_target=False)
    assert not report.ok and "Colonnes manquantes" in report.errors[0]


def test_retard_target_is_balanced_enough(project):
    cfg = load_config("retard_paiement")
    data.generate(cfg)
    rate = data.load(cfg)[cfg.target].mean()
    assert 0.2 < rate < 0.6

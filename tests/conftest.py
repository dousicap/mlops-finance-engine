"""Chaque test tourne dans un projet temporaire : copie des cas d'usage, MLflow isolé."""

import shutil
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]


USECASES = ("defaut_credit", "diabete", "imputation_comptable", "maintenance_predictive", "retard_paiement")
# Jeux réels : échantillon stratifié, trois candidats, porte qualité abaissée à l'échelle du test.
REAL_DATA = {"defaut_credit": 0.6, "maintenance_predictive": 0.3}


def _shrink(root: Path):
    """Jeux de données réduits pour garder des tests rapides."""
    sizes = {"retard_paiement": 3000, "imputation_comptable": 2500,
             "defaut_credit": 3000, "maintenance_predictive": 3000}
    for name, n in sizes.items():
        cfg_file = root / "usecases" / name / "config.yaml"
        cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
        cfg["data"]["n_rows"] = n
        cfg["cv_folds"] = 3
        if name in REAL_DATA:
            cfg["candidates"] = cfg["candidates"][:3]
            cfg["min_score"] = REAL_DATA[name]
        cfg_file.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")


@pytest.fixture
def project(tmp_path, monkeypatch):
    shutil.copytree(REPO / "usecases", tmp_path / "usecases")
    _shrink(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FINML_ROOT", str(tmp_path))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.delenv("FINML_MODELS_DIR", raising=False)
    return tmp_path


def set_config(project: Path, usecase: str, **changes):
    cfg_file = project / "usecases" / usecase / "config.yaml"
    cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    cfg.update(changes)
    cfg_file.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")


@pytest.fixture(scope="session")
def trained_base(tmp_path_factory):
    """Entraîne et exporte tous les cas d'usage une seule fois pour toute la session."""
    from finml import data
    from finml.config import load_config
    from finml.export import export_champion
    from finml.train import train

    base = tmp_path_factory.mktemp("base")
    shutil.copytree(REPO / "usecases", base / "usecases")
    _shrink(base)
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(base)
        mp.setenv("FINML_ROOT", str(base))
        mp.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{base / 'mlflow.db'}")
        for name in USECASES:
            if load_config(name).data.generator:  # sinon le CSV versionné sert directement
                data.generate(load_config(name))
            train(name)
            export_champion(name, base / "models")
    return base


@pytest.fixture
def trained(project, trained_base):
    """Copie isolée du projet entraîné : chaque test a sa base MLflow et ses journaux."""
    for item in ("models", "data"):
        shutil.copytree(trained_base / item, project / item)
    shutil.copy(trained_base / "mlflow.db", project / "mlflow.db")
    return project

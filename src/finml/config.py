"""Chargement et validation de la configuration d'un cas d'usage."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


def project_root() -> Path:
    """Racine du projet : variable FINML_ROOT, sinon le répertoire courant."""
    return Path(os.environ.get("FINML_ROOT", Path.cwd())).resolve()


def usecases_root() -> Path:
    return project_root() / "usecases"


class ColumnRule(BaseModel):
    """Règle de qualité sur une colonne : bornes et valeurs autorisées."""

    min: float | None = None
    max: float | None = None
    allowed: list[Any] | None = None
    nullable: bool = False


class Features(BaseModel):
    numeric: list[str] = Field(default_factory=list)
    categorical: list[str] = Field(default_factory=list)
    text: str | None = None

    @property
    def all(self) -> list[str]:
        cols = [*self.numeric, *self.categorical]
        if self.text:
            cols.append(self.text)
        return cols


class Candidate(BaseModel):
    """Un modèle candidat. Le moteur entraîne tous les candidats et garde le meilleur."""

    name: str
    type: Literal["logistic_regression", "random_forest", "xgboost", "knn_retrieval", "tfidf_logreg"]
    params: dict[str, Any] = Field(default_factory=dict)


class DataConfig(BaseModel):
    generator: str | None = None  # fichier Python exposant generate(n_rows, seed) -> DataFrame
    path: str  # CSV relatif à la racine du dépôt
    n_rows: int = 5000
    seed: int = 42


class UseCaseConfig(BaseModel):
    name: str
    description: str
    task: Literal["tabular_classification", "text_classification"]
    registered_model: str
    target: str
    positive_label: Any | None = None
    features: Features
    schema_rules: dict[str, ColumnRule] = Field(default_factory=dict, alias="schema")
    data: DataConfig
    metric: Literal["roc_auc", "f1", "accuracy", "top3_accuracy", "average_precision"]
    min_score: float = Field(ge=0, le=1)
    test_size: float = 0.2
    candidates: list[Candidate]
    decision_threshold: float = 0.5
    labels: dict[str, str] = Field(default_factory=dict)  # libellés lisibles des classes

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _check_consistency(self) -> UseCaseConfig:
        if self.task == "text_classification" and not self.features.text:
            raise ValueError("Un cas text_classification doit déclarer features.text")
        if self.target in self.features.all:
            raise ValueError("La cible ne peut pas être une variable d'entrée")
        if not self.candidates:
            raise ValueError("Au moins un modèle candidat est requis")
        return self


def usecase_dir(name: str, base: Path | None = None) -> Path:
    return (base or usecases_root()) / name


def load_config(name: str, base: Path | None = None) -> UseCaseConfig:
    path = usecase_dir(name, base) / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Cas d'usage inconnu : {name} ({path} absent)")
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return UseCaseConfig.model_validate(raw)


def list_usecases(base: Path | None = None) -> list[str]:
    root = base or usecases_root()
    return sorted(p.name for p in root.iterdir() if (p / "config.yaml").exists())

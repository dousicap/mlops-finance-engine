"""Génération, chargement et contrôle qualité des données d'un cas d'usage."""

from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from finml.config import UseCaseConfig, project_root, usecase_dir


class DataQualityError(ValueError):
    """Levée quand les données ne respectent pas le contrat déclaré dans la configuration."""


@dataclass
class QualityReport:
    n_rows: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def data_path(cfg: UseCaseConfig) -> Path:
    return project_root() / cfg.data.path


def generate(cfg: UseCaseConfig) -> Path:
    """Exécute le générateur déclaré et écrit le CSV. Déterministe pour une graine donnée."""
    if not cfg.data.generator:
        raise ValueError(f"{cfg.name} : aucun générateur déclaré, fournir le CSV à {cfg.data.path}")
    gen_file = usecase_dir(cfg.name) / cfg.data.generator
    spec = importlib.util.spec_from_file_location(f"finml_gen_{cfg.name}", gen_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    df = module.generate(n_rows=cfg.data.n_rows, seed=cfg.data.seed)
    out = data_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def load(cfg: UseCaseConfig) -> pd.DataFrame:
    path = data_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"Données absentes : {path}. Lancer `finml generate {cfg.name}`.")
    dtype = {cfg.target: str} if cfg.task == "text_classification" else None
    return pd.read_csv(path, dtype=dtype)


def fingerprint(df: pd.DataFrame) -> str:
    """Empreinte SHA-256 du jeu de données, enregistrée dans MLflow pour la traçabilité."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()


def validate(df: pd.DataFrame, cfg: UseCaseConfig, *, with_target: bool = True) -> QualityReport:
    """Vérifie le contrat de données : colonnes présentes, valeurs manquantes, bornes, modalités."""
    report = QualityReport(n_rows=len(df))
    required = list(cfg.features.all) + ([cfg.target] if with_target else [])
    if with_target and cfg.fairness:
        required += cfg.fairness.sensitive  # nécessaires à l'audit, jamais demandées à l'API
    missing = [c for c in required if c not in df.columns]
    if missing:
        report.errors.append(f"Colonnes manquantes : {missing}")
        return report

    df = df.copy()
    for col in cfg.features.numeric:
        coerced = pd.to_numeric(df[col], errors="coerce")
        n_bad = int((coerced.isna() & df[col].notna()).sum())
        if n_bad:
            report.errors.append(f"{col} : {n_bad} valeur(s) non numérique(s)")
        df[col] = coerced

    for col, rule in cfg.schema_rules.items():
        if col not in df.columns:
            continue
        series = df[col]
        n_null = int(series.isna().sum())
        if n_null and not rule.nullable:
            report.errors.append(f"{col} : {n_null} valeur(s) manquante(s)")
        values = series.dropna()
        if rule.min is not None and (values < rule.min).any():
            report.errors.append(f"{col} : {(values < rule.min).sum()} valeur(s) < {rule.min}")
        if rule.max is not None and (values > rule.max).any():
            report.errors.append(f"{col} : {(values > rule.max).sum()} valeur(s) > {rule.max}")
        if rule.allowed is not None:
            bad = sorted(set(values.unique()) - set(rule.allowed), key=str)
            if bad:
                report.errors.append(f"{col} : modalités non autorisées {bad[:5]}")

    if with_target:
        counts = df[cfg.target].value_counts()
        if len(counts) < 2:
            report.errors.append("La cible n'a qu'une seule classe")
        elif counts.min() < 5:
            report.warnings.append(f"Classe très rare dans la cible : {counts.idxmin()} ({counts.min()})")
        if len(df) < 200:
            report.warnings.append(f"Jeu de données réduit : {len(df)} lignes")
    return report


def load_validated(cfg: UseCaseConfig) -> pd.DataFrame:
    df = load(cfg)
    report = validate(df, cfg)
    if not report.ok:
        raise DataQualityError("; ".join(report.errors))
    return df

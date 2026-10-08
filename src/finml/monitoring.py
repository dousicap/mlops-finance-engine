"""Surveillance en production : journal des prédictions, détection de dérive, simulation de trafic.

Journal : une ligne JSON par prédiction, écrite dans FINML_PREDICTION_LOG_DIR/<cas>.jsonl
et envoyée sur la sortie standard (collectée par CloudWatch Logs sur ECS).

Dérive : on compare la production à l'échantillon de référence embarqué avec le modèle
(entrées du jeu de test et sorties du modèle sur ce jeu).
- variables numériques : test de Kolmogorov-Smirnov. Une variable dérive si la p-value passe
  sous alpha corrigé de Bonferroni (plusieurs tests) ET si la statistique KS dépasse un seuil
  d'effet. Avec beaucoup de lignes, la p-value seule signale des écarts sans importance.
- variables catégorielles : Population Stability Index (PSI), dérive au-delà de 0,2.
- texte : longueur des libellés, traitée comme une variable numérique.
- sorties du modèle (probabilité, confiance) : même test, signal précoce de dérive de concept.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from finml.config import UseCaseConfig

log = logging.getLogger("finml.predictions")


class _StdoutHandler(logging.StreamHandler):
    """Écrit toujours dans le sys.stdout courant (capturable en test, redirigeable)."""

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, _value):
        pass


def _stdout_logger() -> logging.Logger:
    """Une ligne JSON brute par prédiction sur stdout : CloudWatch Logs la collecte sur ECS."""
    if not log.handlers:
        handler = _StdoutHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False
    return log
OUTPUT_COLUMNS = ("probabilite", "confiance")
MIN_ROWS = 100


def text_length_column(cfg: UseCaseConfig) -> str:
    return f"longueur_{cfg.features.text}"


class PredictionLogger:
    def __init__(self, log_dir: str | Path | None = None):
        self.log_dir = Path(log_dir or os.environ.get("FINML_PREDICTION_LOG_DIR", "logs"))
        self._lock = threading.Lock()
        self._stdout = _stdout_logger()

    def path(self, usecase: str) -> Path:
        return self.log_dir / f"{usecase}.jsonl"

    def write(self, usecase: str, model_version, inputs: list[dict], outputs: list[dict], latency_ms: float):
        ts = datetime.now(UTC).isoformat()
        lines = []
        for inp, out in zip(inputs, outputs):
            record = {
                "ts": ts,
                "usecase": usecase,
                "model_version": model_version,
                "latency_ms": latency_ms,
                "inputs": inp,
                "outputs": {k: v for k, v in out.items() if k not in ("top3", "exemples_proches")},
            }
            lines.append(json.dumps(record, ensure_ascii=False, default=str))
        for line in lines:
            self._stdout.info(line)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path(usecase).open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def load_production_log(path: str | Path) -> pd.DataFrame:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                rows.append({**rec["inputs"], **rec["outputs"], "ts": rec["ts"]})
    return pd.DataFrame(rows)


def _psi(ref: pd.Series, prod: pd.Series, eps: float = 1e-4) -> float:
    cats = sorted(set(ref.dropna().astype(str)) | set(prod.dropna().astype(str)))
    r = ref.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy() + eps
    p = prod.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy() + eps
    return float(np.sum((p - r) * np.log(p / r)))


def _numeric_columns(cfg: UseCaseConfig, df: pd.DataFrame) -> list[str]:
    cols = list(cfg.features.numeric)
    if cfg.features.text:
        cols.append(text_length_column(cfg))
    return cols + [c for c in OUTPUT_COLUMNS if c in df.columns]


def _with_derived(cfg: UseCaseConfig, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if cfg.features.text and cfg.features.text in df:
        df[text_length_column(cfg)] = df[cfg.features.text].astype(str).str.len()
    return df


def detect_drift(
    cfg: UseCaseConfig,
    reference: pd.DataFrame,
    production: pd.DataFrame,
    alpha: float = 0.05,
    ks_min: float = 0.1,
    psi_max: float = 0.2,
) -> dict:
    reference, production = _with_derived(cfg, reference), _with_derived(cfg, production)
    report = {
        "usecase": cfg.name,
        "genere_le": datetime.now(UTC).isoformat(),
        "lignes_reference": len(reference),
        "lignes_production": len(production),
        "parametres": {"alpha": alpha, "ks_min": ks_min, "psi_max": psi_max},
        "variables": {},
    }
    if len(production) < MIN_ROWS:
        report["statut"] = "donnees_insuffisantes"
        report["derive"] = False
        return report

    numeric = [c for c in _numeric_columns(cfg, reference) if c in production]
    categorical = [c for c in cfg.features.categorical if c in production]
    alpha_corr = alpha / max(len(numeric), 1)  # correction de Bonferroni
    report["parametres"]["alpha_corrige"] = alpha_corr

    for col in numeric:
        ref = pd.to_numeric(reference[col], errors="coerce").dropna()
        prod = pd.to_numeric(production[col], errors="coerce").dropna()
        stat, p = ks_2samp(ref, prod)
        report["variables"][col] = {
            "type": "sortie" if col in OUTPUT_COLUMNS else "numerique",
            "moyenne_reference": round(float(ref.mean()), 4),
            "moyenne_production": round(float(prod.mean()), 4),
            "ks": round(float(stat), 4),
            "p_value": float(p),
            "derive": bool(p < alpha_corr and stat >= ks_min),
        }
    for col in categorical:
        psi = _psi(reference[col], production[col])
        report["variables"][col] = {
            "type": "categorielle",
            "psi": round(psi, 4),
            "derive": bool(psi > psi_max),
        }

    drifted = [c for c, v in report["variables"].items() if v["derive"]]
    report["variables_en_derive"] = drifted
    report["derive"] = bool(drifted)
    report["statut"] = "derive" if drifted else "stable"
    return report


def simulate_records(cfg: UseCaseConfig, n: int, drift: bool, seed: int = 2026) -> list[dict]:
    """Trafic synthétique tiré du générateur du cas d'usage, avec ou sans dérive.

    Dérive simulée : montants et compteurs en hausse, une modalité qui domine (tabulaire),
    ou libellés courts et inexploitables (texte). Les valeurs restent dans le contrat de
    données : l'API les accepte, seule la distribution change.
    """
    import importlib.util

    from finml.config import usecase_dir

    gen_file = usecase_dir(cfg.name) / cfg.data.generator
    spec = importlib.util.spec_from_file_location(f"finml_sim_{cfg.name}", gen_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Tirage dans une population large : les variables propres à un client (ancienneté,
    # historique de retards) ne doivent pas venir d'une poignée de clients. Sinon les lignes
    # ne sont plus indépendantes et le test KS signale une dérive qui n'existe pas.
    pool = module.generate(n_rows=max(n * 20, 5000), seed=seed)
    df = pool.sample(n, random_state=seed).reset_index(drop=True)
    rng = np.random.default_rng(seed)

    if drift and cfg.task == "tabular_classification":
        for col in cfg.features.numeric:
            rule = cfg.schema_rules.get(col)
            if rule and rule.allowed is not None:  # variable codée (0/1) : on ne la décale pas
                continue
            shifted = df[col] * 1.6
            if rule and rule.max is not None:
                shifted = shifted.clip(upper=rule.max)
            df[col] = shifted.round(2) if df[col].dtype.kind == "f" else shifted.round().astype(int)
        for col in cfg.features.categorical:
            allowed = cfg.schema_rules[col].allowed if col in cfg.schema_rules else None
            dominant = allowed[0] if allowed else df[col].iloc[0]
            mask = rng.random(len(df)) < 0.6
            df.loc[mask, col] = dominant
    elif drift:
        mask = rng.random(len(df)) < 0.5
        df.loc[mask, cfg.features.text] = [f"CB {int(x)}" for x in rng.integers(1000, 99999, mask.sum())]

    return df[cfg.features.all].to_dict(orient="records")


def send_traffic(base_url: str, usecase: str, records: list[dict], batch: int = 50) -> int:
    import httpx

    sent = 0
    with httpx.Client(base_url=base_url, timeout=30) as client:
        for i in range(0, len(records), batch):
            r = client.post(f"/predict/{usecase}", json={"records": records[i:i + batch]})
            r.raise_for_status()
            sent += len(records[i:i + batch])
            time.sleep(0.01)
    return sent

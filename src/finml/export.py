"""Export du modèle champion hors du Registry, pour l'embarquer dans l'image Docker.

L'API charge alors le modèle depuis le disque : l'image est autonome et reproductible,
et la version servie est tracée dans manifest.json (version, run, score, empreinte des données).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from finml.config import load_config
from finml.train import CHAMPION, default_tracking_uri


def export_champion(name: str, out_dir: Path) -> Path:
    cfg = load_config(name)
    mlflow.set_tracking_uri(default_tracking_uri())
    client = MlflowClient()
    mv = client.get_model_version_by_alias(cfg.registered_model, CHAMPION)
    run = client.get_run(mv.run_id)
    parent_id = run.data.tags.get("mlflow.parentRunId")
    parent = client.get_run(parent_id) if parent_id else run

    target = out_dir / name
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{cfg.registered_model}@{CHAMPION}",
                                        dst_path=str(target))
    manifest = {
        "usecase": name,
        "registered_model": cfg.registered_model,
        "version": mv.version,
        "alias": CHAMPION,
        "run_id": mv.run_id,
        "candidate": run.info.run_name,
        "metric": cfg.metric,
        "score": run.data.metrics.get(cfg.metric),
        "metier": {k: v for k, v in run.data.metrics.items() if k.startswith("metier_")},
        "equite": {k: v for k, v in run.data.metrics.items() if k.startswith("equite_")},
        "data_sha256": parent.data.tags.get("data_sha256"),
        "git_sha": parent.data.tags.get("git_sha"),
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return target

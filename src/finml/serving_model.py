"""Enveloppe MLflow chargée depuis son code source (« models from code », MLflow 3).

MLflow enregistre ce fichier tel quel au lieu de sérialiser avec pickle l'objet Python. Au chargement,
load_context lit la configuration (JSON) et l'estimateur (skops, types contrôlés).
"""

import json
from pathlib import Path

import mlflow
import pandas as pd

from finml.config import UseCaseConfig
from finml.models import load_estimator, predict_frame


class FinmlModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        cfg_raw = json.loads(Path(context.artifacts["config"]).read_text(encoding="utf-8"))
        self.cfg = UseCaseConfig.model_validate(cfg_raw)
        self.estimator = load_estimator(context.artifacts["estimator"])

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        return predict_frame(self.estimator, self.cfg, model_input)


mlflow.models.set_model(FinmlModel())

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


# Modèles disponibles par type de tâche. Un même nom (ex. random_forest) désigne la version
# classification ou régression selon la tâche du cas d'usage.
CLASSIFICATION_MODELS = {
    "logistic_regression", "naive_bayes", "knn", "svm", "decision_tree", "random_forest",
    "extra_trees", "hist_gradient_boosting", "xgboost", "lightgbm", "mlp", "torch_mlp",
}
REGRESSION_MODELS = {
    "linear_regression", "ridge", "lasso", "elastic_net", "svr", "knn", "decision_tree",
    "random_forest", "extra_trees", "hist_gradient_boosting", "xgboost", "lightgbm", "mlp",
}
TEXT_MODELS = {"tfidf_logreg", "tfidf_svm", "tfidf_naive_bayes", "knn_retrieval"}
MODELS_BY_TASK = {
    "tabular_classification": CLASSIFICATION_MODELS,
    "tabular_regression": REGRESSION_MODELS,
    "text_classification": TEXT_MODELS,
}
# Métriques où plus bas vaut mieux : la porte qualité devient un plafond (max_score).
LOWER_IS_BETTER = {"rmse", "mae"}
METRICS_BY_TASK = {
    "tabular_classification": {"roc_auc", "f1", "accuracy", "average_precision"},
    "tabular_regression": {"rmse", "mae", "r2"},
    "text_classification": {"accuracy", "f1", "top3_accuracy"},
}


class SearchParam(BaseModel):
    """Plage d'un hyperparamètre : entier, réel (échelle log possible) ou liste de choix."""

    int_range: tuple[int, int] | None = Field(default=None, alias="int")
    float_range: tuple[float, float] | None = Field(default=None, alias="float")
    choice: list[Any] | None = None
    log: bool = False

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _one_kind(self) -> SearchParam:
        if sum(x is not None for x in (self.int_range, self.float_range, self.choice)) != 1:
            raise ValueError("Déclarer exactement un de : int, float, choice")
        return self


class Candidate(BaseModel):
    """Un modèle candidat. Le moteur entraîne tous les candidats et garde le meilleur.

    `params` : réglage de départ. `search` : plages explorées par Optuna si le cas d'usage
    déclare un bloc `tuning`. Le réglage trouvé n'est retenu que s'il bat le réglage de départ
    en validation croisée.
    """

    name: str
    type: Literal[
        "logistic_regression", "naive_bayes", "knn", "svm", "decision_tree", "random_forest",
        "extra_trees", "hist_gradient_boosting", "xgboost", "lightgbm", "mlp", "torch_mlp",
        "linear_regression", "ridge", "lasso", "elastic_net", "svr",
        "tfidf_logreg", "tfidf_svm", "tfidf_naive_bayes", "knn_retrieval",
    ]
    params: dict[str, Any] = Field(default_factory=dict)
    search: dict[str, SearchParam] = Field(default_factory=dict)


class TuningConfig(BaseModel):
    """Budget de la recherche d'hyperparamètres, par candidat. Graine : celle des données."""

    n_trials: int = Field(default=20, ge=1)
    timeout_seconds: float = Field(default=60, gt=0)


class SelectionConfig(BaseModel):
    """Règle de choix du champion.

    Deux candidats sont à égalité si le test t corrigé ne montre pas d'écart (p > alpha) ET si
    l'écart de moyenne en validation croisée reste sous la tolérance. À égalité, le modèle le
    plus simple gagne : plus facile à expliquer, plus rapide, plus stable.
    """

    prefer_simpler: bool = True
    alpha: float = Field(default=0.05, gt=0, lt=1)
    tolerance: float = Field(default=0.01, ge=0)


class OutputConfig(BaseModel):
    """Noms des champs de sortie d'une classification multi-classes (API, journal, LLM)."""

    class_field: str = "classe"
    class_name_field: str = "intitule"


class CostBusiness(BaseModel):
    """Classification binaire : le seuil de décision minimise le coût métier attendu."""

    kind: Literal["cost"]
    cost_false_negative: float = Field(gt=0, description="Coût d'un cas positif non détecté")
    cost_false_positive: float = Field(ge=0, description="Coût d'une action déclenchée à tort")
    annual_volume: int = Field(gt=0, description="Nombre de cas traités par an")
    capacity: float | None = Field(default=None, gt=0, le=1,
                                   description="Part maximale de cas que l'équipe peut traiter")
    unit: str = "EUR"


class AutomationBusiness(BaseModel):
    """Classification multi-classes : seuil de confiance qui garantit une précision cible."""

    kind: Literal["automation"]
    target_precision: float = Field(gt=0, le=1)
    annual_volume: int = Field(gt=0)
    minutes_per_item: float = Field(gt=0, description="Temps de traitement manuel d'un élément")


class FairnessConfig(BaseModel):
    """Audit d'équité d'une décision binaire (système à haut risque au sens de l'AI Act).

    Les colonnes sensibles sont présentes dans les données mais ne sont jamais des variables
    du modèle : elles servent seulement à mesurer si la décision traite les groupes de la même
    façon. Une alerte ne bloque pas l'entraînement : elle demande une décision humaine.
    """

    sensitive: list[str] = Field(min_length=1)
    max_recall_gap: float = Field(default=0.10, gt=0, le=1, description="Égalité des chances")
    min_favorable_ratio: float = Field(default=0.80, gt=0, le=1, description="Règle des quatre cinquièmes")
    min_group_size: int = Field(default=100, ge=1, description="Effectif minimal pour comparer un groupe")
    min_group_positives: int = Field(default=30, ge=1, description="Cas positifs minimaux pour comparer un rappel")


class DataConfig(BaseModel):
    generator: str | None = None  # fichier Python exposant generate(n_rows, seed) -> DataFrame
    path: str  # CSV relatif à la racine du dépôt
    n_rows: int = 5000
    seed: int = 42


class LLMConfig(BaseModel):
    """Second avis par LLM sur les cas envoyés en revue humaine (cas d'usage texte).

    Le LLM ne décide jamais : sa proposition est montrée au comptable, avec sa justification.
    Toute API compatible OpenAI convient. Mistral par défaut, pour un hébergement européen.
    """

    prompt: str  # chemin du prompt versionné, relatif au dossier du cas d'usage
    prompt_version: str
    base_url: str = "https://api.mistral.ai/v1"
    model: str = "mistral-small-latest"
    api_key_env: str = "FINML_LLM_API_KEY"
    temperature: float = 0.0
    timeout_seconds: float = 20.0
    # Tarifs en EUR par million de tokens : à renseigner avec la grille du fournisseur retenu.
    price_input_per_mtok: float = 0.0
    price_output_per_mtok: float = 0.0


class UseCaseConfig(BaseModel):
    name: str
    description: str
    task: Literal["tabular_classification", "tabular_regression", "text_classification"]
    registered_model: str
    target: str
    positive_label: Any | None = None
    features: Features
    schema_rules: dict[str, ColumnRule] = Field(default_factory=dict, alias="schema")
    data: DataConfig
    metric: Literal["roc_auc", "f1", "accuracy", "top3_accuracy", "average_precision", "rmse", "mae", "r2"]
    min_score: float | None = Field(default=None, le=1)  # porte qualité, métriques où plus haut vaut mieux
    max_score: float | None = Field(default=None, ge=0)  # porte qualité, métriques d'erreur (rmse, mae)
    test_size: float = 0.2
    candidates: list[Candidate]
    decision_threshold: float = 0.5  # recalculé à l'entraînement quand un bloc business existe
    labels: dict[str, str] = Field(default_factory=dict)  # libellés lisibles des classes
    cv_folds: int = Field(default=5, ge=2, le=10)  # validation croisée pour départager les candidats
    business: CostBusiness | AutomationBusiness | None = Field(default=None, discriminator="kind")
    llm: LLMConfig | None = None
    fairness: FairnessConfig | None = None
    output: OutputConfig = Field(default_factory=OutputConfig)
    tuning: TuningConfig | None = None
    selection: SelectionConfig = Field(default_factory=SelectionConfig)

    model_config = {"populate_by_name": True}

    @property
    def is_regression(self) -> bool:
        return self.task == "tabular_regression"

    @property
    def is_tabular(self) -> bool:
        return self.task != "text_classification"

    def better(self, a: float, b: float) -> bool:
        """a est-il strictement meilleur que b pour la métrique du cas d'usage ?"""
        return a < b if self.metric in LOWER_IS_BETTER else a > b

    def passes_gate(self, score: float) -> bool:
        if self.metric in LOWER_IS_BETTER:
            return score <= self.max_score
        return score >= self.min_score

    @model_validator(mode="after")
    def _check_consistency(self) -> UseCaseConfig:
        if self.task == "text_classification" and not self.features.text:
            raise ValueError("Un cas text_classification doit déclarer features.text")
        if self.metric not in METRICS_BY_TASK[self.task]:
            raise ValueError(f"Métrique {self.metric} incompatible avec {self.task}")
        if self.metric in LOWER_IS_BETTER and self.max_score is None:
            raise ValueError(f"{self.metric} : déclarer max_score (erreur maximale acceptée)")
        if self.metric not in LOWER_IS_BETTER and self.min_score is None:
            raise ValueError(f"{self.metric} : déclarer min_score (score minimal accepté)")
        wrong = [c.name for c in self.candidates if c.type not in MODELS_BY_TASK[self.task]]
        if wrong:
            raise ValueError(f"Candidats incompatibles avec {self.task} : {wrong}")
        if self.task == "tabular_regression" and (self.business or self.fairness):
            raise ValueError("Blocs business et fairness : réservés à la classification")
        if self.target in self.features.all:
            raise ValueError("La cible ne peut pas être une variable d'entrée")
        if not self.candidates:
            raise ValueError("Au moins un modèle candidat est requis")
        if len({c.name for c in self.candidates}) != len(self.candidates):
            raise ValueError("Deux candidats portent le même nom")
        if self.business and self.business.kind == "cost" and self.task != "tabular_classification":
            raise ValueError("Le bloc business « cost » s'applique à une classification binaire tabulaire")
        if self.llm and self.task != "text_classification":
            raise ValueError("Le second avis LLM s'applique aux cas d'usage texte")
        if self.business and self.business.kind == "automation" and self.task == "tabular_regression":
            raise ValueError("Le bloc business « automation » s'applique à une classification multi-classes")
        if self.fairness:
            if self.task != "tabular_classification":
                raise ValueError("L'audit d'équité s'applique à une décision binaire tabulaire")
            leaked = set(self.fairness.sensitive) & set(self.features.all + [self.target])
            if leaked:
                raise ValueError(f"Colonnes sensibles utilisées comme variables ou cible : {sorted(leaked)}")
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

"""Préparation du jeu réel « Default of Credit Card Clients » (UCI, Taïwan, 2005).

Source et licence : voir source/SOURCE.md. Le fichier brut est versionné, son empreinte est
vérifiée avant chaque préparation : une source modifiée arrête la chaîne.

Choix de préparation :
- colonnes renommées en français, mois 1 = mois le plus récent (septembre 2005) ;
- codes d'éducation non documentés (0, 5, 6) regroupés avec « autre » (4) : 345 lignes ;
- situation familiale 0, non documentée, regroupée avec « autre » (3) : 54 lignes ;
- sexe, âge et situation familiale sont conservés pour l'audit d'équité mais ne sont PAS des
  variables du modèle (voir config.yaml, bloc fairness) ;
- montants en dollars taïwanais (NT$), tels que publiés.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

SOURCE = Path(__file__).parent / "source" / "default_of_credit_card_clients.csv.gz"
SHA256 = "fad968254e68183c43ebbe35adfa588c0e5073a816b811495646f511d79a0472"

EDUCATION = {1: "master", 2: "licence", 3: "lycee", 4: "autre", 0: "autre", 5: "autre", 6: "autre"}
MARRIAGE = {1: "marie", 2: "celibataire", 3: "autre", 0: "autre"}
SEX = {1: "homme", 2: "femme"}
MONTHS = ["0", "2", "3", "4", "5", "6"]  # PAY_0 puis PAY_2 à PAY_6 : pas de PAY_1 dans la source


def _check_source() -> None:
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"Source modifiée : {SOURCE.name} a l'empreinte {digest}, attendu {SHA256}")


def _age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, bins=[0, 29, 39, 49, 200], labels=["moins_30", "30-39", "40-49", "50_plus"]).astype(str)


def generate(n_rows: int, seed: int) -> pd.DataFrame:
    """Renvoie le jeu préparé. n_rows < 30 000 : échantillon stratifié (tests, simulation)."""
    _check_source()
    raw = pd.read_csv(SOURCE)
    df = pd.DataFrame({
        "limite_credit": raw["LIMIT_BAL"],
        "niveau_etudes": raw["EDUCATION"].map(EDUCATION),
        **{f"statut_paiement_m{i + 1}": raw[f"PAY_{m}"] for i, m in enumerate(MONTHS)},
        **{f"facture_m{i}": raw[f"BILL_AMT{i}"] for i in range(1, 7)},
        **{f"paiement_m{i}": raw[f"PAY_AMT{i}"] for i in range(1, 7)},
        # Audit d'équité uniquement
        "sexe": raw["SEX"].map(SEX),
        "tranche_age": _age_band(raw["AGE"]),
        "situation_familiale": raw["MARRIAGE"].map(MARRIAGE),
        "defaut_mois_suivant": raw["default payment next month"],
    })
    if n_rows < len(df):
        df = (df.groupby("defaut_mois_suivant", group_keys=False)
                .sample(frac=n_rows / len(df), random_state=seed)
                .sample(frac=1, random_state=seed))
    return df.reset_index(drop=True)

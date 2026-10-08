"""Préparation du jeu réel « AI4I 2020 Predictive Maintenance » (UCI).

Source et licence : voir source/SOURCE.md. Le fichier brut est versionné, son empreinte est
vérifiée avant chaque préparation.

Choix de préparation :
- les colonnes TWF, HDF, PWF, OSF et RNF (mode de la panne) sont SUPPRIMÉES : elles ne sont
  connues qu'après la panne. Les garder donnerait un score presque parfait et un modèle
  inutilisable en production (fuite de la cible) ;
- les identifiants (UDI, Product ID) sont supprimés : ils n'ont pas de sens physique ;
- 27 lignes ont un indicateur de panne incohérent avec les modes déclarés (panne sans mode, ou
  mode sans panne). Elles sont gardées telles quelles : c'est le bruit réel des étiquettes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

SOURCE = Path(__file__).parent / "source" / "ai4i2020.csv.gz"
SHA256 = "ef58eba4050205298dd15cc05b2169a810eef82f31a8d64b51fe5d923dffb537"


def _check_source() -> None:
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"Source modifiée : {SOURCE.name} a l'empreinte {digest}, attendu {SHA256}")


def generate(n_rows: int, seed: int) -> pd.DataFrame:
    """Renvoie le jeu préparé. n_rows < 10 000 : échantillon stratifié (tests, simulation)."""
    _check_source()
    raw = pd.read_csv(SOURCE)
    df = pd.DataFrame({
        "gamme": raw["Type"],
        "temperature_air_k": raw["Air temperature [K]"],
        "temperature_process_k": raw["Process temperature [K]"],
        "vitesse_rotation_tr_min": raw["Rotational speed [rpm]"],
        "couple_nm": raw["Torque [Nm]"],
        "usure_outil_min": raw["Tool wear [min]"],
        "panne": raw["Machine failure"],
    })
    if n_rows < len(df):
        df = (df.groupby("panne", group_keys=False)
                .sample(frac=n_rows / len(df), random_state=seed)
                .sample(frac=1, random_state=seed))
    return df.reset_index(drop=True)

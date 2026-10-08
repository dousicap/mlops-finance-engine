"""Factures clients fictives, avec un risque de retard de paiement réaliste.

Le retard dépend de l'historique du client, de son secteur, de sa taille, du mode de paiement,
de la saison et du délai contractuel. Un bruit important est ajouté : le problème n'est pas
trivialement séparable, comme dans la réalité.
"""

import numpy as np
import pandas as pd

SECTEURS = {"btp": 0.55, "commerce": 0.15, "services": 0.0, "industrie": 0.1, "sante": -0.2, "public": 0.35}
TAILLES = {"tpe": 0.35, "pme": 0.1, "eti": -0.15, "ge": -0.05}
MODES = {"prelevement": -1.1, "virement": 0.0, "lcr": -0.3, "cheque": 0.45}


def generate(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_clients = max(n_rows // 6, 50)
    client_profil = rng.normal(0, 0.8, n_clients)  # comportement de paiement propre au client
    client_secteur = rng.choice(list(SECTEURS), n_clients, p=[0.18, 0.2, 0.3, 0.15, 0.07, 0.1])
    client_taille = rng.choice(list(TAILLES), n_clients, p=[0.35, 0.4, 0.18, 0.07])
    client_anciennete = rng.integers(1, 121, n_clients)

    cid = rng.integers(0, n_clients, n_rows)
    secteur = client_secteur[cid]
    taille = client_taille[cid]
    anciennete = client_anciennete[cid]
    mode = rng.choice(list(MODES), n_rows, p=[0.25, 0.55, 0.08, 0.12])
    delai = rng.choice([30, 45, 60], n_rows, p=[0.5, 0.3, 0.2])
    mois = rng.integers(1, 13, n_rows)
    montant = np.round(np.exp(rng.normal(7.3, 1.1, n_rows)), 2).clip(50, 250000)

    tendance = client_profil[cid]
    nb_retards_12m = rng.poisson(np.exp(0.6 + 0.9 * tendance).clip(0.1, 12))
    retard_moyen = np.where(
        nb_retards_12m > 0, np.round(rng.gamma(2.0, 6 + 6 * np.exp(tendance))), 0
    ).clip(0, 120)
    relance_auto = rng.random(n_rows) < 0.45

    logit = (
        -1.0
        + 1.1 * tendance
        + 0.18 * np.log1p(nb_retards_12m)
        + 0.012 * retard_moyen
        + np.vectorize(SECTEURS.get)(secteur)
        + np.vectorize(TAILLES.get)(taille)
        + np.vectorize(MODES.get)(mode)
        + 0.15 * np.log10(montant)
        - 0.006 * anciennete
        + 0.004 * (delai - 30)
        + np.where(np.isin(mois, [7, 8, 12]), 0.35, 0.0)
        - 0.4 * relance_auto
        + rng.normal(0, 0.9, n_rows)
    )
    retard = (rng.random(n_rows) < 1 / (1 + np.exp(-logit))).astype(int)

    return pd.DataFrame(
        {
            "montant_ttc": montant,
            "delai_contractuel_jours": delai,
            "mois_emission": mois,
            "anciennete_client_mois": anciennete,
            "nb_retards_12m": nb_retards_12m,
            "retard_moyen_12m_jours": retard_moyen,
            "relance_automatique": relance_auto.astype(int),
            "secteur": secteur,
            "taille_client": taille,
            "mode_paiement": mode,
            "paye_en_retard": retard,
        }
    )

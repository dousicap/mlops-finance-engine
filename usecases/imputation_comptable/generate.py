"""Libellés de pièces d'achat fictifs, imputés sur un compte de charge du PCG.

Les libellés imitent ce qu'on lit sur un relevé bancaire ou une facture fournisseur :
majuscules, abréviations, références, fautes de frappe, et quelques libellés ambigus
(un « abonnement » peut être des télécoms ou une location de logiciel). Tous les noms
de fournisseurs sont inventés.
"""

import numpy as np
import pandas as pd

COMPTES = {
    "6061": (["electricite", "fourniture energie", "facture gaz", "eau consommation", "conso elec",
              "carburant gazole", "plein carburant", "electricite bureaux"],
             ["Voltis Energie", "Hydrea Eaux", "GazNord", "Station Relais 7"]),
    "6063": (["petit outillage", "produits entretien", "ampoules et visserie", "aspirateur",
              "petit materiel", "consommables atelier", "produits menage"],
             ["Bricomax Pro", "Hygiena", "OutiPlus"]),
    "6064": (["fournitures bureau", "ramettes papier", "cartouches encre", "toner imprimante",
              "classeurs stylos", "papeterie", "enveloppes"],
             ["Bureau Direct", "Papeteria", "OfficeLine"]),
    "6132": (["loyer bureaux", "loyer local commercial", "loyer mensuel", "quittance loyer",
              "charges locatives bureau", "loyer entrepot"],
             ["SCI Les Tilleuls", "Fonciere Atlas", "SCI Horizon"]),
    "6135": (["location vehicule", "location copieur", "location materiel", "leasing photocopieur",
              "location nacelle", "abonnement logiciel saas", "licence logiciel mensuelle"],
             ["LocaPro", "Copitech Location", "CloudSoft", "Rentauto"]),
    "6155": (["reparation vehicule", "entretien vehicule", "revision utilitaire", "maintenance copieur",
              "reparation machine", "contrat maintenance"],
             ["Garage Central", "Meca Services", "Copitech Maintenance"]),
    "616": (["prime assurance", "assurance multirisque", "assurance flotte auto", "cotisation assurance rc",
             "echeance assurance", "assurance locaux"],
            ["Mutuelle Atlantique", "Assurances Delta", "Protectis"]),
    "6226": (["honoraires expert comptable", "honoraires avocat", "honoraires conseil",
              "mission commissaire aux comptes", "honoraires juridiques", "note honoraires"],
             ["Cabinet Durand Associes", "Lexia Avocats", "Audit Conseil Ouest"]),
    "6231": (["annonce publicitaire", "insertion annonce legale", "publicite en ligne", "campagne ads",
              "annonce recrutement", "parution journal"],
             ["MediaPlus", "Annonces Legales 75", "Clicads"]),
    "6251": (["billet train", "billet avion", "taxi", "peage autoroute", "hotel deplacement",
              "frais kilometriques", "parking aeroport", "nuitee hotel"],
             ["RailExpress", "AirLigne", "Hotel du Parc", "TaxiCity"]),
    "6257": (["repas client", "restaurant client", "dejeuner affaires", "reception clients",
              "traiteur reunion", "diner partenaires"],
             ["Brasserie du Port", "Le Bistrot", "Traiteur Saveurs"]),
    "626": (["abonnement telephone", "forfait mobile", "abonnement internet", "frais postaux",
             "affranchissement courrier", "box fibre", "lettre recommandee"],
            ["TeleCom One", "FibreNet", "La Poste Express"]),
    "627": (["frais bancaires", "commission tenue compte", "frais carte bancaire", "commission virement",
             "agios", "cotisation carte", "frais terminal paiement"],
            ["Banque Ouest", "Credit Rhodanien", "NeoBanque"]),
    "6185": (["inscription salon", "seminaire formation", "conference professionnelle", "colloque",
              "participation congres"],
             ["Expo Events", "Forum Pro", "Congres Sud"]),
}

PREFIXES = ["", "", "", "PRLV SEPA ", "CB ", "VIR ", "FACT ", "FAC N° "]

# Fournisseurs multi-activités : le nom seul ne suffit pas à trouver le compte.
FOURNISSEURS_MIXTES = {
    "MarketPro": ["6063", "6064"],
    "Banque Ouest": ["627", "616"],
    "Garage Central": ["6155", "6061"],
    "TeleCom One": ["626", "6135"],
    "Hotel du Parc": ["6251", "6257", "6185"],
}
TAUX_ERREUR_HISTORIQUE = 0.03  # imputations erronées présentes dans l'historique du cabinet


def _bruit(texte: str, rng: np.random.Generator) -> str:
    if rng.random() < 0.08 and len(texte) > 6:  # faute de frappe : lettre supprimée
        i = int(rng.integers(1, len(texte) - 1))
        texte = texte[:i] + texte[i + 1:]
    if rng.random() < 0.5:
        texte = texte.upper()
    return texte


def generate(n_rows: int = 4000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    comptes = list(COMPTES)
    poids = np.array([1.4, 0.8, 1.0, 0.9, 1.0, 0.8, 0.6, 0.7, 0.5, 1.6, 1.0, 1.0, 1.3, 0.3])
    tirage = rng.choice(comptes, n_rows, p=poids / poids.sum())
    lignes = []
    for compte in tirage:
        motifs, fournisseurs = COMPTES[compte]
        motif = motifs[int(rng.integers(len(motifs)))]
        mixtes = [f for f, cs in FOURNISSEURS_MIXTES.items() if compte in cs]
        if mixtes and rng.random() < 0.35:
            fournisseur = mixtes[int(rng.integers(len(mixtes)))]
        else:
            fournisseur = fournisseurs[int(rng.integers(len(fournisseurs)))]
        prefixe = PREFIXES[int(rng.integers(len(PREFIXES)))]
        ref = f" {int(rng.integers(1000, 99999))}" if rng.random() < 0.5 else ""
        mois = f" {['JANV', 'FEVR', 'MARS', 'AVR', 'MAI', 'JUIN', 'JUIL', 'AOUT', 'SEPT', 'OCT', 'NOV', 'DEC'][int(rng.integers(12))]}" if rng.random() < 0.3 else ""
        ordre = rng.random()
        if ordre < 0.4:
            libelle = f"{prefixe}{fournisseur} {motif}{mois}{ref}"
        elif ordre < 0.75:
            libelle = f"{prefixe}{motif} {fournisseur}{ref}"
        elif ordre < 0.95:
            libelle = f"{prefixe}{fournisseur}{ref}"  # libellé pauvre : seul le fournisseur aide
        else:
            libelle = f"{prefixe or 'CB '}{int(rng.integers(1000, 99999))}"  # libellé inexploitable
        if rng.random() < TAUX_ERREUR_HISTORIQUE:
            compte = comptes[int(rng.integers(len(comptes)))]
        lignes.append({"libelle": _bruit(libelle.strip(), rng), "compte": compte})
    return pd.DataFrame(lignes)

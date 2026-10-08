# Spec : maintenance prédictive d'une ligne d'usinage

## 1. Décision métier
À chaque cycle, la machine reçoit une probabilité de panne. Au-dessus du seuil, un technicien
intervient avant le cycle suivant (changement d'outil, contrôle). Sans modèle, la maintenance
est seulement curative.

Utilisateurs : responsable maintenance, techniciens. Moment : en continu, sur les relevés capteurs.

## 2. Valeur et coûts
| Élément | Hypothèse | Justification |
|---|---|---|
| Panne non anticipée | 3 000 EUR | arrêt de ligne, réparation curative, pièces en urgence |
| Intervention inutile | 250 EUR | technicien, arrêt planifié court |
| Volume | 50 000 cycles contrôlés par an | |

Rapport de coût de 12 pour 1 : le seuil optimal est très bas. Mesuré à 0,0088, il était hors
d'atteinte de l'ancienne grille de seuils (0,05 à 0,95). Le moteur cherche désormais le seuil
exact parmi les scores observés.

## 3. Données
Source publique : UCI « AI4I 2020 Predictive Maintenance », 10 000 relevés, licence CC BY 4.0
(`usecases/maintenance_predictive/source/SOURCE.md`). Jeu synthétique publié par son auteur pour
reproduire des données industrielles : référence publique, pas mesure d'un site réel.

Variables : gamme de produit (L, M, H), température de l'air et du process, vitesse de rotation,
couple, usure de l'outil. Cible : panne (3,4 % des cycles).

**Fuite de la cible écartée.** Les colonnes TWF, HDF, PWF, OSF et RNF indiquent le mode de la
panne. Elles ne sont connues qu'après la panne : `prepare.py` les supprime, un test le vérifie.
27 lignes ont un indicateur de panne incohérent avec les modes : gardées, c'est le bruit réel.

## 4. Modèles mis en concurrence
8 candidats sur l'average precision, validation croisée à 5 plis : régression logistique,
k plus proches voisins, random forest, extra trees, gradient boosting par histogramme, XGBoost,
LightGBM, réseau de neurones PyTorch. L'average precision remplace l'AUC : avec 3,4 % de pannes,
l'AUC reste flatteuse même quand le classement des cas rares est mauvais.

## 5. Critères d'acceptation et résultats (données publiques)
| Critère | Exigence | Mesuré |
|---|---|---|
| Average precision sur le jeu de test | ≥ 0,60 | 0,855 (gradient boosting par histogramme) |
| Cycles signalés | | 6,7 % |
| Pannes anticipées (rappel) | | 87 % |
| Précision des interventions | | 44 % |
| Coût par cycle | | 22,88 EUR contre 102 EUR en curatif seul |
| Gain annuel | > 0 | 3 956 250 EUR sur 50 000 cycles |

La régression logistique tombe à 0,469 en CV, contre 0,802 pour le boosting : les pannes suivent
des règles à seuil (dissipation thermique, puissance, surcharge outil), qu'un modèle linéaire ne
représente pas. C'est l'inverse du cas `retard_paiement`, où le linéaire gagne.

## 6. Exploitation
API `POST /predict/maintenance_predictive` appelée par le système de supervision. Le contrat
borne chaque mesure physique : une valeur hors bornes (capteur défaillant) est refusée (422)
au lieu de produire une prédiction fausse. Dérive surveillée sur l'usure et le couple.

## 7. Risques
Les probabilités du boosting ne sont pas calibrées : le seuil de 0,0088 n'est pas une
probabilité réelle de panne, c'est un rang. À recalculer à chaque réentraînement. Sur un site
réel, le coût d'une panne varie selon la machine : prévoir un coût par gamme.

# Spec : détection précoce du défaut sur cartes de crédit

## 1. Décision métier
Une fois par an, chaque titulaire de carte reçoit une probabilité de défaut au mois suivant.
Les dossiers les plus risqués, dans la limite de la capacité de l'équipe, passent en
surveillance renforcée : contact, plan de paiement, ajustement de limite. Sans modèle,
aucune surveillance préventive n'est faite.

Utilisateurs : équipe risque et recouvrement d'un émetteur de cartes. Moment : revue annuelle
du portefeuille.

## 2. Valeur et coûts
| Élément | Hypothèse | Justification |
|---|---|---|
| Défaut non anticipé | 500 EUR | part de la perte récupérable si la surveillance démarre un mois plus tôt |
| Surveillance inutile | 60 EUR | temps d'analyste, marge perdue sur une limite réduite, risque d'attrition |
| Volume | 30 000 cartes, une revue par an | |
| Capacité | 20 % du portefeuille, soit 6 000 dossiers | taille de l'équipe |

**Pourquoi une capacité.** Sans plafond, le seuil de coût minimal signale 67 % des clients
(mesuré). Aucune équipe ne traite ce volume, et la comparaison se fait alors avec « surveiller
tout le monde » : gain de 163 400 EUR par an seulement. Avec le plafond, la seule politique
réalisable sans modèle est de ne surveiller personne.

## 3. Données
Source réelle : UCI « Default of Credit Card Clients », 30 000 clients, Taïwan, avril à
septembre 2005, licence CC BY 4.0 (`usecases/defaut_credit/source/SOURCE.md`). Fichier versionné,
empreinte SHA-256 vérifiée avant chaque préparation.

Variables du modèle : limite de crédit, niveau d'études, statut de paiement, montant facturé et
montant payé sur 6 mois. Cible : défaut le mois suivant (22 % des clients, taux de crise).

Qualité constatée et traitée dans `prepare.py` :
- 345 codes d'éducation non documentés (0, 5, 6), regroupés dans « autre » ;
- 54 situations familiales codées 0, non documentées, regroupées dans « autre » ;
- le statut de paiement -2 n'est pas décrit par la source. Lecture retenue : aucune utilisation.

**Exclues du modèle et gardées pour l'audit :** sexe, tranche d'âge, situation familiale.

## 4. Modèles mis en concurrence
6 candidats sur l'AUC, validation croisée à 5 plis : régression logistique, random forest,
gradient boosting par histogramme, XGBoost, LightGBM, réseau de neurones PyTorch (servi en ONNX).

## 5. Critères d'acceptation et résultats (données réelles)
| Critère | Exigence | Mesuré |
|---|---|---|
| AUC sur le jeu de test | ≥ 0,72 | 0,776 (random forest) |
| Dossiers signalés | ≤ 20 % | 19,8 %, seuil 0,33 |
| Défauts captés (rappel) | | 50 % |
| Précision des signalements | | 56 % |
| Coût par client | | 59,91 EUR contre 110,58 EUR sans surveillance |
| Gain annuel | > 0 | 1 520 200 EUR |
| Écart de rappel entre groupes | ≤ 0,10 | sexe 0,012, âge 0,060, situation familiale 0,010 |
| Ratio de taux favorable | ≥ 0,80 | sexe 0,967, âge 0,959, situation familiale 0,989 |

Les cinq modèles non linéaires sont à égalité (0,780 à 0,783 en CV, écart-type 0,004 à 0,006).
La régression logistique décroche (0,727) : la relation entre retards et défaut n'est pas additive.

## 6. Audit d'équité (AI Act)
Le scoring de crédit est un système à haut risque (AI Act, annexe III). À chaque entraînement,
le moteur mesure par groupe le taux signalé, le rappel et la précision au seuil servi. Il trace
les écarts dans MLflow (`equite_*`, `equite.json`) et met le statut en tag. Une alerte ne bloque
pas le déploiement : elle exige une revue humaine documentée.

Deux constats mesurés :
- le groupe « situation familiale autre » compte 75 clients et 12 défauts. Son rappel (0,33)
  déclenchait une alerte. Il est maintenant décrit mais exclu des comparaisons (moins de 100
  clients ou de 30 défauts) ;
- sans plafond de capacité, le ratio de taux favorable par âge tombe à 0,75 : alerte. La même
  forêt aléatoire, avec un seuil plus haut, passe à 0,96. L'équité dépend de la politique de
  décision, pas seulement du modèle.

## 7. Exploitation
API `POST /predict/defaut_credit` ou lot `finml batch`. Les colonnes sensibles ne sont jamais
demandées à l'API. Dérive surveillée sur les statuts de paiement et les montants. Réentraînement
à chaque revue, avec nouvel audit d'équité.

## 8. Risques
Données de 2005, en période de crise : un portefeuille actuel aura un taux de défaut bien plus
bas, donc un autre seuil. Les variables autorisées peuvent porter un biais indirect (la limite de
crédit dépend de l'âge) : c'est ce que l'audit mesure à chaque version.

# Spec de cas d'usage : <nom>

Une spec par cas d'usage, validée avec le métier avant toute ligne de code. Elle devient
`usecases/<nom>/config.yaml` : chaque section ci-dessous a son champ dans la configuration.

## 1. Décision métier
- Quelle décision le modèle prépare-t-il, pour qui, à quel moment ?
- Que se passe-t-il aujourd'hui sans modèle (politique de référence) ?

## 2. Valeur et coûts (`business`)
- Coût d'une erreur de chaque type, en euros ou en minutes, avec la source du chiffre.
- Volume annuel traité.
- Règle de décision : seuil au coût minimal (binaire) ou précision cible (automatisation).

## 3. Données (`data`, `features`, `schema`)
- Source, fréquence, volume, période d'historique.
- Variables autorisées. Exclure toute variable connue seulement après la décision (fuite).
- Contrat : type, bornes, modalités, valeurs manquantes acceptées ou non.

## 4. Modèles mis en concurrence (`candidates`)
- Au moins une référence simple (régression logistique ou naive Bayes) et des modèles plus riches.
- Sélection sur la moyenne en validation croisée, mesure finale sur un jeu de test jamais vu.

## 5. Critères d'acceptation
- Métrique technique minimale (`min_score`) : en dessous, rien n'est déployé.
- Gain métier minimal par rapport à la politique de référence.
- Latence maximale par appel, taux de revue humaine acceptable.

## 6. Exploitation
- Qui consomme la prédiction (API, lot nocturne), qui traite les cas en revue.
- Surveillance : dérive des entrées et des sorties, alertes, fréquence de réentraînement.
- Données personnelles et conservation (RGPD), traçabilité attendue par l'audit.

## 7. Risques
- Effets d'un faux positif et d'un faux négatif sur le client final.
- Ce qui change si une hypothèse de coût est fausse (analyse de sensibilité).

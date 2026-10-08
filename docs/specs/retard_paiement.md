# Spec : prévision des retards de paiement clients

## 1. Décision métier
Chaque facture émise reçoit une probabilité de retard. Au-dessus du seuil, un chargé de
recouvrement appelle le client avant l'échéance. Sans modèle, l'équipe relance soit toutes
les factures, soit aucune avant échéance.

Utilisateurs : responsable du poste clients, chargés de recouvrement. Moment : à l'émission.

## 2. Valeur et coûts
| Élément | Hypothèse | Justification |
|---|---|---|
| Retard non anticipé | 40 EUR | 45 jours de financement du besoin de trésorerie, relance tardive manuelle, part de risque d'impayé |
| Relance préventive | 15 EUR | environ 15 minutes d'un chargé de recouvrement |
| Volume | 8 000 factures par an | |

Règle : le seuil de décision minimise le coût attendu. Il est choisi sur les prédictions
hors échantillon de la validation croisée, puis mesuré sur le jeu de test.

**Sensibilité mesurée.** Le modèle ne crée de valeur que si une relance coûte plus d'environ
un quart d'un retard. À 80 EUR le retard et 4 EUR la relance, relancer toutes les factures
reste la meilleure politique et le modèle n'apporte presque rien (112 EUR par an).

## 3. Données
Historique de factures : montant, délai contractuel, mois d'émission, ancienneté du client,
nombre et durée moyenne des retards sur 12 mois, relance automatique, secteur, taille du
client, mode de paiement. Cible : payée après l'échéance (0/1).

Exclu : date de paiement réelle, relances déjà faites après l'émission (connues trop tard).
Contrat de données : `usecases/retard_paiement/config.yaml`, section `schema`.

## 4. Modèles mis en concurrence
11 candidats, validation croisée à 5 plis sur l'AUC : régression logistique, naive Bayes,
k plus proches voisins, SVM linéaire calibré, random forest, extra trees, gradient boosting
par histogramme, XGBoost, LightGBM, réseau de neurones scikit-learn, réseau de neurones
PyTorch (servi en ONNX).

## 5. Critères d'acceptation et résultats (données de démonstration)
| Critère | Exigence | Mesuré |
|---|---|---|
| AUC sur le jeu de test | ≥ 0,70 | 0,741 (régression logistique) |
| Gain annuel face à la meilleure politique sans modèle | > 0 | 11 025 EUR |
| Factures signalées | | 60 %, seuil 0,32 |
| Rappel sur les retards | | 79 % |
| Latence par appel | < 100 ms | 36 ms en conteneur |

Le modèle linéaire gagne parce que le signal est surtout additif. Les modèles d'arbres et
les réseaux de neurones n'apportent rien ici, à complexité bien supérieure.

## 6. Exploitation
API `POST /predict/retard_paiement` à l'émission, ou lot nocturne `finml batch`. Dérive
surveillée sur les montants, l'historique de retards et la probabilité moyenne. Réentraînement
mensuel, ou sur alerte de dérive.

## 7. Risques
Un faux positif gêne un bon payeur : limiter le ton de la relance préventive. Les hypothèses de
coût dominent le résultat : les faire valider par la direction financière avant mise en service.

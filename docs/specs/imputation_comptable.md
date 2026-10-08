# Spec : imputation comptable automatique des pièces d'achat

## 1. Décision métier
Pour chaque pièce d'achat (libellé bancaire ou facture fournisseur), proposer le compte de
charge du PCG. Si la confiance est suffisante, l'écriture est passée automatiquement. Sinon,
la pièce part en revue chez le comptable, avec le top 3 du modèle, les écritures passées les
plus proches et, si un LLM est configuré, un second avis justifié.

Utilisateurs : collaborateurs comptables d'un cabinet. Moment : à l'import des relevés et factures.

## 2. Valeur et coûts
| Élément | Hypothèse |
|---|---|
| Volume | 24 000 pièces d'achat par an |
| Saisie manuelle | 1,5 minute par pièce |
| Précision exigée sur les écritures automatiques | 97 % |

L'historique contient environ 3 % d'erreurs d'imputation. Une cible de 98 % n'automatise
presque rien, car la précision est mesurée contre ces étiquettes imparfaites. Nettoyer
l'historique est le premier levier de gain.

## 3. Données
Libellé de la pièce et compte retenu dans l'historique du cabinet. 14 comptes de charge du
PCG (classe 6). Libellés courts, abréviations, fautes de frappe, fournisseurs multi-activités.

## 4. Modèles mis en concurrence
5 candidats sur l'exactitude, validation croisée à 5 plis : TF-IDF de n-grammes de caractères
avec régression logistique, SVM linéaire calibré ou naive Bayes, et recherche vectorielle des
plus proches voisins (k = 5 et 15). Des embeddings sémantiques multilingues ont été mesurés à
part : 0,83 contre 0,88, car la forme du libellé compte plus que son sens.

## 5. Critères d'acceptation et résultats (données de démonstration)
| Critère | Exigence | Mesuré |
|---|---|---|
| Exactitude sur le jeu de test | ≥ 0,80 | 0,885 (TF-IDF + naive Bayes) |
| Bon compte dans le top 3 | | 0,937 |
| Pièces passées automatiquement | | 85 % |
| Précision des écritures automatiques | ≥ 97 % | 97 % |
| Temps économisé | | 508 heures par an |

## 6. Second avis LLM (LLMOps)
Pour les 15 % de pièces en revue : appel d'une API compatible OpenAI (Mistral par défaut,
hébergement européen), prompt versionné (`prompts/second_avis_v1.md`), sortie JSON contrainte
à la liste des comptes autorisés. Toute réponse hors liste ou mal formée est rejetée. Chaque
appel est tracé dans MLflow : version du prompt, modèle, tokens, latence, coût. Le LLM ne passe
jamais d'écriture : il éclaire la décision du comptable.

Évaluation : `finml llm-eval imputation_comptable` compare, sur les pièces en revue, l'exactitude
du LLM à celle du modèle seul. Un nouveau prompt n'est adopté que s'il fait mieux.

## 7. Risques
Une mauvaise imputation automatique fausse les comptes du client : seuil de précision
conservateur et contrôle par échantillonnage mensuel. Les libellés peuvent contenir des données
personnelles (noms de salariés) : hébergement européen et durée de conservation définie.

# finml : automatiser des décisions métier avec MLOps et LLMOps

Moteur de machine learning industrialisé une seule fois, réutilisé pour chaque décision métier à
automatiser : direction financière, cabinet comptable, risque de crédit, maintenance industrielle. Chaque cas d'usage part d'une
spec (décision, valeur, critères d'acceptation), se déclare dans un dossier (configuration +
données), et le moteur fait le reste :

1. contrôle qualité des données (contrat déclaré) ;
2. tournoi de modèles en validation croisée, du plus simple au réseau de neurones ;
3. calibrage du seuil de décision sur la valeur métier (euros, heures), sous contrainte de
   capacité de l'équipe, pas sur un score ;
4. audit d'équité pour les décisions à haut risque (AI Act) ;
5. suivi MLflow, porte qualité, promotion champion / challenger dans le Registry ;
6. API, conteneur, déploiement AWS (ECS) ou Kubernetes ;
7. surveillance : métriques, journal des prédictions, dérive, alertes ;
8. second avis LLM tracé et évalué pour les cas que le modèle ne tranche pas.

> Deux cas tournent sur des jeux publics UCI (licence CC BY 4.0), versionnés avec leur empreinte
> SHA-256 : `defaut_credit` (données réelles, 30 000 clients) et `maintenance_predictive`
> (10 000 relevés, jeu de référence synthétique publié par son auteur). Les deux cas comptables
> utilisent des données générées par code, avec du bruit réaliste. Les hypothèses de coût sont
> déclarées dans chaque configuration et restent à valider avec le métier.

## Cas d'usage livrés

| Cas d'usage | Décision automatisée | Champion | Score test | Valeur métier mesurée |
|---|---|---|---|---|
| `retard_paiement` | Quelles factures relancer avant échéance ? | Régression logistique | ROC AUC 0,741 | 11 575 € par an sur 8 000 factures |
| `imputation_comptable` | Sur quel compte du PCG passer cette pièce d'achat ? | TF-IDF + naive Bayes | Exactitude 0,885, top-3 0,937 | 85 % des pièces passées seules à 97 % de précision, 508 h par an |
| `defaut_credit` | Quels titulaires de carte surveiller avant un défaut ? | Random forest | ROC AUC 0,776 | Coût par client de 110,58 € à 59,91 €, 6 000 dossiers au plus, audit d'équité ok |
| `maintenance_predictive` | Quelle machine arrêter avant la panne ? | Gradient boosting par histogramme | Average precision 0,855 | 87 % des pannes anticipées, coût par cycle de 102 € à 22,88 € |

Chaque cas a sa spec dans [`docs/specs/`](docs/specs) : décision, hypothèses de valeur, données,
modèles, critères d'acceptation, exploitation, risques. Le gabarit est
[`docs/specs/_modele.md`](docs/specs/_modele.md).

## Tournoi de modèles

Tous les candidats déclarés passent la même validation croisée stratifiée à 5 plis. Le champion
est choisi sur la moyenne de CV, puis mesuré une seule fois sur un jeu de test tenu à l'écart.
Le classement complet est enregistré dans MLflow (`classement.csv`) et affiché par `finml train`.

**retard_paiement** (ROC AUC, 11 candidats)

| Candidat | CV moyenne | Écart-type | Test |
|---|---|---|---|
| **Régression logistique** | **0,736** | 0,011 | **0,741** |
| SVM linéaire calibré | 0,736 | 0,011 | 0,740 |
| Réseau de neurones PyTorch (servi en ONNX) | 0,732 | 0,009 | 0,735 |
| Random forest | 0,729 | 0,009 | 0,732 |
| Réseau de neurones scikit-learn | 0,725 | 0,020 | 0,731 |
| Extra trees | 0,724 | 0,006 | 0,736 |
| XGBoost | 0,722 | 0,011 | 0,725 |
| Gradient boosting histogramme | 0,720 | 0,009 | 0,726 |
| LightGBM | 0,717 | 0,011 | 0,720 |
| Naive Bayes | 0,706 | 0,017 | 0,719 |
| k plus proches voisins | 0,702 | 0,006 | 0,692 |

**defaut_credit** (ROC AUC, 6 candidats, données réelles)

| Candidat | CV moyenne | Écart-type | Test |
|---|---|---|---|
| **Random forest** | **0,783** | 0,004 | **0,776** |
| XGBoost | 0,781 | 0,006 | 0,774 |
| Gradient boosting histogramme | 0,780 | 0,004 | 0,777 |
| LightGBM | 0,780 | 0,006 | 0,773 |
| Réseau de neurones PyTorch (servi en ONNX) | 0,780 | 0,006 | 0,771 |
| Régression logistique | 0,727 | 0,010 | 0,707 |

**maintenance_predictive** (average precision, 8 candidats, 3,4 % de pannes)

| Candidat | CV moyenne | Écart-type | Test |
|---|---|---|---|
| **Gradient boosting histogramme** | **0,802** | 0,038 | **0,855** |
| LightGBM | 0,792 | 0,040 | 0,847 |
| Random forest | 0,772 | 0,021 | 0,787 |
| XGBoost | 0,769 | 0,042 | 0,856 |
| Réseau de neurones PyTorch (servi en ONNX) | 0,692 | 0,037 | 0,698 |
| Extra trees | 0,692 | 0,026 | 0,669 |
| k plus proches voisins | 0,517 | 0,024 | 0,486 |
| Régression logistique | 0,469 | 0,032 | 0,421 |

**imputation_comptable** (exactitude, 5 candidats)

| Candidat | CV moyenne | Écart-type | Test |
|---|---|---|---|
| **TF-IDF n-grammes de caractères + naive Bayes** | **0,905** | 0,009 | **0,885** |
| TF-IDF + SVM linéaire calibré | 0,904 | 0,010 | 0,885 |
| Recherche vectorielle k=15 (TF-IDF réduit à 256 dimensions) | 0,904 | 0,006 | 0,889 |
| TF-IDF + régression logistique | 0,901 | 0,012 | 0,888 |
| Recherche vectorielle k=5 | 0,897 | 0,010 | 0,888 |

Ce que le tournoi montre :
- Le signal du retard de paiement est surtout linéaire : la régression logistique bat les arbres
  boostés et le réseau PyTorch. Le moteur choisit sur la mesure, pas sur la réputation du modèle,
  et garde le modèle le plus simple à expliquer à un DAF.
- En maintenance, c'est l'inverse : la régression logistique tombe à 0,47 contre 0,80. Les pannes
  suivent des règles à seuil (dissipation thermique, puissance, surcharge de l'outil).
- Sur le jeu de maintenance, XGBoost a le meilleur test (0,856) mais pas la meilleure CV : il
  n'est pas retenu. Choisir sur le test, c'est sur-apprendre le jeu de test.
- Les écarts entre les cinq premiers sont inférieurs à un écart-type de CV : ils sont à égalité.
  C'est pourquoi la sélection se fait sur la CV et non sur le test, qui ne sert qu'à confirmer.
- Sur des libellés bancaires courts et bruités, les n-grammes de caractères battent des embeddings
  sémantiques multilingues (0,83, mesure séparée) : noms de fournisseurs et fautes de frappe
  comptent plus que le sens.
- Le réseau PyTorch est entraîné avec arrêt précoce, exporté en ONNX et servi par onnxruntime :
  pas de dépendance PyTorch dans l'image de production. Sans PyTorch installé, ce candidat est
  ignoré proprement et tracé (`candidats_ignores`).

## De la prédiction à la décision : couche métier

Un score ne dit pas quoi faire. Chaque configuration déclare un bloc `business`, et le moteur
calibre le seuil servi sur les prédictions hors échantillon de la validation croisée.

**Coût (`kind: cost`).** Le seuil retenu minimise le coût attendu. Il est cherché exactement
parmi les scores observés, pas sur une grille, et se compare aux politiques sans modèle
réalisables. Un plafond `capacity` limite la part de cas signalés à ce que l'équipe peut traiter.

| Cas | Coût FN / FP | Seuil | Part signalée | Rappel | Coût par cas | Sans modèle | Économie annuelle |
|---|---|---|---|---|---|---|---|
| retard_paiement | 40 € / 15 € | 0,29 | 67 % | 85 % | 7,47 € | 8,92 € | 11 575 € |
| defaut_credit | 500 € / 60 €, capacité 20 % | 0,33 | 20 % | 50 % | 59,91 € | 110,58 € | 1 520 200 € |
| maintenance_predictive | 3 000 € / 250 € | 0,0088 | 6,7 % | 87 % | 22,88 € | 102 € | 3 956 250 € |

Trois points d'entretien, tous mesurés :
- `retard_paiement` avec 80 € / 4 € : le modèle fait perdre 560 € par an, car relancer tout le
  monde coûte moins cher. Le modèle ne crée de la valeur que si une relance coûte plus d'environ
  un quart d'un retard. Cette sensibilité se vérifie avec le métier avant tout développement.
- `defaut_credit` sans plafond : le seuil optimal signale 67 % des clients, intenable pour une
  équipe. Le plafond de 20 % rend la décision exploitable.
- `maintenance_predictive` : avec un coût de 12 pour 1, le seuil optimal (0,0088) était hors de
  la grille 0,05-0,95 utilisée au départ. D'où la recherche exacte.

**Automatisation (`kind: automation`, imputation_comptable).** Le seuil est la confiance la plus
basse qui garde 97 % de précision sur les écritures passées automatiquement. Résultat : 85 % des
24 000 pièces annuelles passées sans intervention, soit 508 heures économisées à 1,5 minute par
pièce. Une cible de 98 % n'automatisait que 3 % des pièces : l'historique contient environ 3 %
d'erreurs d'étiquetage, et nettoyer cet historique est le premier levier de gain.

Les métriques `metier_*` sont tracées dans MLflow et embarquées dans `manifest.json`.

## Audit d'équité (AI Act)

Le scoring de crédit est un système à haut risque (AI Act, annexe III). Le bloc `fairness`
déclare des colonnes sensibles. Elles sont présentes dans les données mais interdites comme
variables du modèle (la configuration est refusée sinon). À chaque entraînement, le moteur mesure
par groupe, au seuil servi : taux signalé, rappel, précision. Il trace dans MLflow l'écart de
rappel et le ratio de taux favorable (règle des quatre cinquièmes), le détail dans `equite.json`
et le statut en tag. Une alerte ne bloque pas le déploiement : elle exige une revue humaine.

| defaut_credit | Écart de rappel (max 0,10) | Ratio de taux favorable (min 0,80) |
|---|---|---|
| Sexe | 0,012 | 0,967 |
| Tranche d'âge | 0,060 | 0,959 |
| Situation familiale | 0,010 | 0,989 |

Deux constats mesurés :
- le groupe « situation familiale autre » compte 75 clients et 12 défauts. Son rappel déclenchait
  une fausse alerte. Les groupes de moins de 100 cas ou 30 positifs sont maintenant décrits mais
  exclus des comparaisons ;
- sans plafond de capacité, le ratio par âge tombe à 0,75 : alerte. Même modèle, autre seuil :
  l'équité dépend de la politique de décision, pas seulement du modèle.

## Second avis LLM (LLMOps)

Les 15 % de pièces sous le seuil partent en revue chez le comptable. Pour elles, un LLM propose
un second avis justifié. Il ne passe jamais d'écriture.

| Exigence | Mise en œuvre |
|---|---|
| Fournisseur interchangeable | API compatible OpenAI (Mistral par défaut, hébergement européen ; vLLM, Ollama ou OpenAI par configuration) |
| Prompt versionné | `usecases/imputation_comptable/prompts/second_avis_v1.md`, version tracée à chaque appel |
| Contexte récupéré | top 3 du modèle et écritures passées les plus proches (recherche vectorielle) |
| Garde-fous | température 0, sortie JSON validée par Pydantic, compte obligatoirement dans la liste autorisée, consigne d'ignorer toute instruction contenue dans le libellé |
| Résilience | une panne ou un délai dépassé chez le fournisseur ne bloque pas la revue (`appel_en_echec`) |
| Observabilité | trace MLflow par appel : prompt, modèle, tokens, latence, coût ; compteur Prometheus `finml_llm_second_avis_total` par résultat |
| Évaluation | `finml llm-eval` compare, sur les pièces en revue, l'exactitude du LLM à celle du modèle seul ; un nouveau prompt n'est adopté que s'il fait mieux |
| Désactivation propre | sans clé d'API, `/second-avis` répond 503 et le reste du service fonctionne |

```bash
export FINML_LLM_API_KEY=...            # clé Mistral (ou autre fournisseur configuré)
finml llm-eval imputation_comptable --n 50
finml llm-eval imputation_comptable --provider echo   # hors ligne, sans clé (CI)
```

Les tests couvrent la chaîne hors ligne (client `echo`) : garde-fous, panne, traces, endpoint.
Aucun appel à un vrai fournisseur n'a été mesuré dans ce dépôt.

## Architecture

```mermaid
flowchart LR
    S[docs/specs<br/>décision, valeur, critères] --> Cas
    subgraph Cas["usecases/&lt;cas&gt;/"]
        C[config.yaml] --- G[generate.py ou CSV]
        C --- PR[prompts/]
    end
    Cas --> Q[Contrôle qualité<br/>contrat de données]
    Q --> T[Tournoi<br/>CV 5 plis, N candidats]
    T --> B[Seuil métier<br/>coût ou automatisation]
    B --> M[(MLflow<br/>runs, classement, metier_*)]
    M --> P{Porte qualité<br/>min_score}
    P -- échec --> X[CI en échec<br/>rien n'est déployé]
    P -- succès --> R[(Registry<br/>champion / challenger)]
    R --> E[Export du champion<br/>+ manifest.json]
    E --> D[Image Docker]
    D --> ECS[ECS Fargate ou Kubernetes]
    ECS --> API[/predict<br/>/second-avis/]
    API --> LLM[LLM compatible OpenAI<br/>traces MLflow]
    ECS --> J[Journal des prédictions] --> DR{finml drift<br/>KS + PSI}
    ECS --> PM[/metrics Prometheus/] --> AL[Alertes Grafana]
    DR -- dérive --> RT[Réentraînement]
```

Chaîne de déploiement (`.github/workflows/deploy.yml`), à chaque push sur `main` :

1. lint et tests (`ruff`, `pytest`) ;
2. génération des données, tournoi, seuil métier, porte qualité, export des champions ;
3. authentification AWS par OIDC, sans clé stockée dans GitHub (étape sautée tant que
   `AWS_DEPLOY_ROLE_ARN` n'est pas défini) ;
4. build de l'image, tag = SHA du commit (tags immuables dans ECR) ;
5. déploiement ECS rolling ou blue/green, retour arrière automatique.

## Démarrage rapide

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install -e ".[torch]" --extra-index-url https://download.pytorch.org/whl/cpu   # optionnel

finml list                  # cas d'usage déclarés
finml run-all               # données -> tournoi -> Registry -> export dans models/
pytest                      # 51 tests

uvicorn finml.api:app --port 8000                     # http://localhost:8000/docs
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://localhost:5000
```

Pièce passée automatiquement :

```bash
curl -s localhost:8000/predict/imputation_comptable -H 'content-type: application/json' \
  -d '{"records":[{"libelle":"CB HOTEL DU PARC NUITEE HOTEL MARS"}]}'
```

```json
{"compte": "6251", "intitule": "Voyages et déplacements", "confiance": 0.9997, "a_revoir": false,
 "top3": [{"compte": "6251", "score": 0.9997}, {"compte": "6257", "score": 0.0001}, ...]}
```

Pièce envoyée en revue, avec second avis :

```bash
curl -s localhost:8000/second-avis/imputation_comptable -H 'content-type: application/json' \
  -d '{"libelle":"PRLV SEPA TeleCom One 4521"}'
```

```json
{"modele": {"compte": "626", "intitule": "Frais postaux et de télécommunications",
            "confiance": 0.4086, "a_revoir": true,
            "top3": [{"compte": "626", "score": 0.4086}, {"compte": "6135", "score": 0.1154}, ...]},
 "second_avis": {"compte": "...", "confiance": "moyenne", "justification": "...", "valide": true,
                 "prompt_version": "second_avis_v1", "latence_ms": ..., "cout_eur": ...}}
```

Prédiction par lots, avec le même contrat de données que l'API :

```bash
finml batch retard_paiement factures.csv predictions.csv
```

Avec Docker (serveur MLflow, pgvector et API) : voir l'en-tête de `docker-compose.yml`.

## Ajouter un cas d'usage

1. Écrire la spec à partir de `docs/specs/_modele.md` : décision, valeur, critères d'acceptation.
2. Créer `usecases/<nom>/config.yaml` : cible, variables, contrat de données, métrique, seuil
   minimal, candidats, bloc `business`, bloc `llm` optionnel.
3. Fournir les données : un script exposant `generate(n_rows, seed)`. Pour un jeu réel, ce script
   (`prepare.py`) lit la source versionnée dans `source/`, vérifie son empreinte SHA-256 et
   prépare les colonnes. Une source modifiée arrête la chaîne.
4. `finml run-all`. L'API expose automatiquement `POST /predict/<nom>`.

Aucune ligne du moteur ne change : `defaut_credit` et `maintenance_predictive` ont été ajoutés
ainsi. Cas suivants prévus de la même façon : détection de fraude sur notes de frais, prévision
de trésorerie, rapprochement bancaire.

## Ce que le moteur garantit

| Garantie | Mécanisme |
|---|---|
| Données conformes | Contrat déclaré dans la configuration, vérifié à l'entraînement **et** à chaque appel API (422 sinon) |
| Source intègre | Jeux réels versionnés, empreinte SHA-256 vérifiée avant chaque préparation, licence et citation dans `source/SOURCE.md` |
| Pas de fuite de la cible | Variables connues après la décision supprimées à la préparation, vérifié par un test |
| Choix de modèle robuste | Tournoi en validation croisée, test tenu à l'écart, classement archivé |
| Seuil justifié | Calculé exactement sur la valeur métier déclarée, sous plafond de capacité, comparé aux politiques sans modèle réalisables |
| Équité auditée | Colonnes sensibles interdites comme variables, écarts par groupe tracés à chaque version |
| Traçabilité | Empreinte SHA-256 des données, SHA Git, paramètres et métriques de chaque candidat dans MLflow |
| Pas de régression | Un nouveau modèle ne devient `champion` que s'il bat l'actuel, sinon il reste `challenger` |
| Pas de modèle faible en production | Porte qualité `min_score` : la CI échoue, rien n'est déployé |
| Pas de trafic vers un conteneur inutilisable | `/health` renvoie 503 tant qu'un modèle attendu n'est pas chargé |
| Image reproductible | Modèles champions embarqués, version servie lisible dans `manifest.json` et `/usecases` |
| Aucun pickle sur le chemin de service | Estimateurs sérialisés avec skops, rechargés seulement si chaque type figure dans une liste blanche, vérifiée par un test sur tous les candidats. Réseau PyTorch servi en ONNX. Enveloppe MLflow « models from code » |
| LLM sous contrôle | Prompt versionné, sortie validée, coût et latence tracés, jamais de décision automatique |
| Dérive détectée | Journal de chaque prédiction, comparaison à l'échantillon de référence embarqué avec le modèle |

## Surveillance en production

- **Métriques Prometheus** sur `/metrics` : prédictions, erreurs par cause, latence, distribution des scores de sortie, version servie (`finml_model_info`). Un seul processus par conteneur pour des compteurs exacts, la montée en charge passe par le nombre de tâches ECS.
- **Journal des prédictions** : une ligne JSON par prédiction (entrées, sorties, version, latence), écrite sur la sortie standard, donc collectée par CloudWatch Logs sur ECS, et dans `logs/<cas>.jsonl` en local.
- **Détection de dérive** : `finml drift <cas>` compare le journal à l'échantillon de référence enregistré avec le modèle (entrées et sorties du jeu de test).

| Type de variable | Test | Dérive si |
|---|---|---|
| Numérique | Kolmogorov-Smirnov | p-value < alpha corrigé (Bonferroni) **et** statistique KS ≥ 0,1 |
| Catégorielle | Population Stability Index | PSI > 0,2 |
| Texte | KS sur la longueur des libellés | idem numérique |
| Sortie du modèle | KS sur la probabilité ou la confiance | idem numérique, signal précoce de dérive de concept |

Le seuil sur la statistique KS évite de déclencher sur des écarts minimes quand le volume est grand.
Code retour 3 en cas de dérive (exploitable par un job planifié), rapport JSON dans `reports/`,
historique dans l'expérience MLflow `finml-<cas>-monitoring`.

Démonstration sur 20 lots de 300 prédictions par cas d'usage :

| Cas d'usage | Fausses alertes (trafic normal) | Dérives détectées (trafic dérivé) |
|---|---|---|
| retard_paiement | 0 / 20 | 20 / 20 |
| imputation_comptable | 0 / 20 | 20 / 20 |

```bash
uvicorn finml.api:app --port 8000 &
finml simulate retard_paiement --n 300            # trafic normal
finml simulate retard_paiement --n 300 --drift    # montants en hausse, une modalité domine
finml drift retard_paiement                       # stable ou dérive, code retour 0 ou 3
```

Piège rencontré et corrigé : une première simulation tirait 300 factures de seulement 50 clients.
Les variables propres au client (ancienneté) n'étaient plus indépendantes d'une ligne à l'autre, et
le test KS signalait une dérive sur du trafic normal. La simulation tire désormais dans une large
population de clients. En production, le même phénomène existe : un gros client qui facture beaucoup
un jour donné peut déclencher une fausse alerte.

## Recherche vectorielle (pgvector)

Le référentiel de libellés déjà imputés est indexé dans PostgreSQL/pgvector (vecteurs denses de
256 dimensions, index HNSW, similarité cosinus). Il permet de retrouver les écritures passées les
plus proches d'un nouveau libellé, pour justifier une proposition.

```bash
docker compose up -d pgvector
finml index-pgvector imputation_comptable \
  --dsn postgresql://finml:finml@localhost:5432/finml \
  --query "PRLV SEPA FIBRENET box internet"
# 5000 vecteurs indexés dans ref_imputation_comptable
#  {'libelle': 'ABONNEMENT INTERNET FIBRENET', 'compte': '626', 'similarite': 0.777}
```

## Observabilité (Prometheus, Grafana)

```bash
docker compose up -d api && docker compose --profile monitoring up -d
# Prometheus http://localhost:9090   Grafana http://localhost:3000 (lecture seule)
```

![Tableau de bord Grafana](docs/img/grafana-finml.png)

Cinq alertes (`deploy/monitoring/alerts.yml`), chacune avec l'action à mener :

| Alerte | Déclenchement | Gravité |
|---|---|---|
| FinmlApiDown | l'API ne répond plus au scrape depuis 1 min | critique |
| FinmlModelErrors | au moins une erreur du modèle sur 5 min | critique |
| FinmlInvalidInputs | plus de 0,05 requête refusée par seconde pendant 10 min | avertissement |
| FinmlLatencyP95High | latence p95 au-dessus de 500 ms pendant 10 min | avertissement |
| FinmlScoreShift | score moyen de l'heure écarté de plus de 0,15 de celui des 7 jours | avertissement, déclenche `finml drift` |

Les alertes ont des tests unitaires (`promtool test rules deploy/monitoring/alerts_test.yml`), exécutés en CI.

## Kubernetes (`deploy/k8s`, Kustomize)

Alternative à ECS pour un client qui exploite déjà un cluster (EKS, AKS, GKE ou sur site).

| Ressource | Choix |
|---|---|
| Deployment | 2 réplicas, rolling update `maxUnavailable: 0`, anti-affinité et répartition sur plusieurs zones |
| Sondes | `startupProbe` jusqu'à 150 s pour charger les modèles, `readinessProbe` sur `/health` (503 sans modèle, donc pas de trafic), `livenessProbe` TCP pour ne pas redémarrer en boucle un pod dont seul un modèle manque |
| Sécurité | namespace en Pod Security « restricted », non-root, système de fichiers en lecture seule, capacités retirées, pas de jeton de service monté, NetworkPolicy (Ingress et Prometheus seulement) |
| Ressources | requêtes 250m / 768 Mi, limites 1 CPU / 1,5 Gi |
| Capacité | HPA de 2 à 6 pods sur le CPU (10 en prod), PodDisruptionBudget |
| Exposition | Service ClusterIP, Ingress nginx avec TLS |

```bash
kubectl apply -k deploy/k8s/overlays/prod
kubectl -n finml rollout status deployment/finml-api
kubectl -n finml rollout undo deployment/finml-api      # retour arrière
```

Vérifications faites : rendu Kustomize, validation stricte contre le schéma Kubernetes 1.31
(kubeconform, 9 ressources valides), aucune alerte kube-linter, et exécution de l'image sous les
contraintes du pod (utilisateur 1001, lecture seule, volumes temporaires, 1,5 Go de mémoire).
Le déploiement sur un vrai cluster n'a pas été exécuté. Le workflow `deploy-k8s.yml` (manuel)
déploie sur EKS et revient en arrière si les pods ne deviennent pas prêts en 5 minutes. Le cluster
EKS lui-même n'est pas créé par ce Terraform, et le rôle de déploiement doit recevoir l'accès au cluster.

## Infrastructure AWS (Terraform, `infra/terraform`)

Région Paris (`eu-west-3`). Ressources : VPC sur 2 zones, sous-réseaux publics (ALB) et privés
(conteneurs, derrière une NAT), security groups (l'API n'accepte que l'ALB), ALB avec HTTPS optionnel,
ECR à tags immuables, cluster ECS Fargate, service avec circuit breaker et autoscaling sur le CPU,
logs CloudWatch, alarme sur l'absence de tâche saine, rôles IAM au moindre privilège, fournisseur
OIDC GitHub limité à la branche `main` du dépôt, secrets lus dans AWS Secrets Manager.

```bash
cd infra/terraform
cp example.tfvars terraform.tfvars    # renseigner github_repository
terraform init && terraform apply
# Puis dans GitHub : Settings > Secrets and variables > Actions > Variables
#   AWS_DEPLOY_ROLE_ARN = <sortie github_deploy_role_arn>
# Le premier push sur main publie l'image et démarre le service.
```

`enable_nat_gateway = false` place les conteneurs en sous-réseaux publics pour une démo à moindre
coût. En production, garder `true`.

### Deux stratégies de déploiement

| | `deployment_strategy = "rolling"` (défaut) | `deployment_strategy = "blue_green"` |
|---|---|---|
| Mécanisme | ECS remplace les tâches progressivement | CodeDeploy démarre une version green complète à côté de blue |
| Exposition | directe | green d'abord visible sur le listener de test :9000, réservé à `test_listener_cidrs` |
| Bascule | au fil du remplacement | canary : 10 % du trafic pendant 5 min, puis 100 % |
| Retour arrière | circuit breaker si les health checks échouent | automatique sur échec **ou sur alarme** (5xx, latence p95), blue gardé 15 min |
| Coût | aucun surcoût | deux versions en parallèle pendant la bascule |

Pour activer le blue/green : `deployment_strategy = "blue_green"` dans `terraform.tfvars`, puis la
variable de dépôt GitHub `DEPLOYMENT_STRATEGY = blue_green`. Le workflow passe alors par CodeDeploy
avec `appspec.yaml`.

## Ce qui a changé en MLOps et LLMOps, et où le dépôt y répond

| Tendance du marché (2025-2026) | Réponse dans finml |
|---|---|
| La valeur métier prime sur le score | Bloc `business`, seuil calibré en euros ou en heures, spec par cas d'usage |
| Spec-driven development, souvent avec un assistant IA | `docs/specs/` : critères d'acceptation écrits avant le code, vérifiés par les tests |
| MLflow 3 unifie ML classique et GenAI | Même serveur pour runs, Registry, traces LLM et évaluations de prompts |
| Prompts traités comme du code | Fichier versionné, version tracée, évaluation avant adoption |
| Observabilité LLM (tokens, coût, latence) | Spans MLflow et compteur Prometheus par résultat |
| Souveraineté et AI Act | Audit d'équité des décisions à haut risque, fournisseur LLM européen par défaut, humain dans la boucle, journal des décisions |
| Données réelles et lignée | Sources versionnées avec empreinte et licence, préparation reproductible |
| Sécurité de la chaîne ML | Pas de pickle (skops, ONNX), OIDC sans clé, images non root, pods « restricted » |
| Portabilité des modèles de deep learning | Entraînement PyTorch, service ONNX Runtime |

## Limites connues et suites

- Les cas comptables tournent sur données générées. `defaut_credit` date de 2005, en pleine crise
  des cartes à Taïwan (22 % de défauts) : un portefeuille actuel demande un autre seuil.
  `maintenance_predictive` est un jeu de référence synthétique. Les hypothèses de coût sont à
  valider avec le métier.
- Les probabilités des modèles boostés ne sont pas calibrées : le seuil de maintenance (0,0088)
  est un rang, pas une probabilité de panne.
- Second avis LLM testé hors ligne seulement (client `echo`). À mesurer avec une vraie clé
  avant de citer un gain.
- Kubernetes validé par rendu, schéma et lint, jamais déployé sur un vrai cluster. Terraform
  validé, `terraform apply` jamais exécuté.
- Le Registry MLflow vit dans la CI (SQLite archivé comme artefact). Étape suivante : un serveur
  MLflow partagé, adossé à RDS et S3.
- La détection de dérive tourne à la demande. Étape suivante : un job planifié qui lance
  `finml drift` et déclenche le réentraînement sur alerte.
- Le vrai signal de dérive de concept arrive avec les étiquettes réelles (date de paiement
  effective, compte corrigé par le comptable), à rapprocher des prédictions journalisées.

## Structure

```
docs/specs/        une spec par cas d'usage + gabarit
src/finml/         moteur : config, data, models, torch_model, vector, train, business,
                   fairness, export, api, llm, monitoring, cli
usecases/          un dossier par cas d'usage : config.yaml, generate.py ou prepare.py,
                   source/ (jeu réel, empreinte, licence), prompts/
tests/             unitaires et bout en bout (génération -> tournoi -> API -> LLM hors ligne)
infra/terraform/   infrastructure AWS (ECS, rolling ou blue/green CodeDeploy)
deploy/k8s/        manifestes Kubernetes (Kustomize, base + overlay prod)
deploy/monitoring/ Prometheus, alertes et leurs tests, Grafana
.github/workflows  CI (lint, tests, Terraform, Kubernetes, alertes), déploiement ECS, déploiement EKS
```

## Stack

Python 3.12, scikit-learn, XGBoost, LightGBM, PyTorch, ONNX Runtime, MLflow 3 (tracking, Registry,
alias, models from code, tracing), skops, FastAPI, Pydantic, API LLM compatible OpenAI (Mistral),
Prometheus, Grafana, SciPy, pgvector, Docker, Kubernetes (Kustomize), GitHub Actions (OIDC),
Terraform, AWS (ECS Fargate, CodeDeploy, ALB, ECR, CloudWatch, IAM, Secrets Manager).

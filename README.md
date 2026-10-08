# finml : un socle MLOps, des cas d'usage financiers

Moteur de machine learning industrialisé une seule fois, réutilisé pour chaque nouveau cas d'usage.
Un cas d'usage se déclare dans un dossier (configuration + données). Le moteur se charge du reste :
contrôle qualité des données, entraînement de plusieurs modèles, suivi MLflow, sélection du meilleur,
promotion dans le Registry, API, conteneur, déploiement sur AWS, puis surveillance en production
(métriques, journal des prédictions, détection de dérive).

L'effort se concentre ainsi sur ce qui crée de la valeur : les données du client et son cas métier.

> Les données sont fictives et générées par code. Les scores mesurent la chaîne, pas une performance
> métier réelle. Brancher un vrai jeu de données revient à remplacer le générateur par un CSV.

## Cas d'usage livrés

| Cas d'usage | Question métier | Type | Meilleur modèle | Score (jeu de test) |
|---|---|---|---|---|
| `retard_paiement` | Cette facture client sera-t-elle payée en retard ? | Tabulaire | Régression logistique | ROC AUC 0,741 |
| `imputation_comptable` | Sur quel compte du PCG imputer ce libellé de pièce ? | Texte | Recherche vectorielle k=15 | Accuracy 0,889, top-3 0,935 |

Comparaison des candidats, enregistrée automatiquement dans MLflow :

| Cas d'usage | Candidat | Score |
|---|---|---|
| retard_paiement | Régression logistique | **0,741** |
| | Random forest | 0,732 |
| | XGBoost | 0,725 |
| imputation_comptable | Recherche vectorielle k=15 (TF-IDF réduit à 256 dimensions) | **0,889** |
| | Recherche vectorielle k=5 | 0,888 |
| | TF-IDF + régression logistique | 0,888 |
| | Embeddings sémantiques MiniLM multilingue, k=5 (mesure séparée) | 0,833 |

Deux constats utiles :
- Sur un signal surtout linéaire, la régression logistique bat XGBoost. Le moteur choisit sur la mesure, pas sur la réputation du modèle.
- Sur des libellés bancaires courts et bruités, les n-grammes de caractères battent les embeddings sémantiques : les noms de fournisseurs et les fautes de frappe comptent plus que le sens.
- À score égal, la recherche vectorielle a un avantage métier : chaque proposition est justifiée par les écritures passées les plus proches (`exemples_proches`), que le comptable peut vérifier.

## Architecture

```mermaid
flowchart LR
    subgraph Cas["usecases/&lt;cas&gt;/"]
        C[config.yaml] --- G[generate.py ou CSV]
    end
    Cas --> Q[Contrôle qualité<br/>schéma, bornes, modalités]
    Q --> T[Entraînement<br/>N candidats]
    T --> M[(MLflow<br/>runs, métriques, artefacts)]
    M --> P{Porte qualité<br/>min_score}
    P -- échec --> X[CI en échec<br/>rien n'est déployé]
    P -- succès --> R[(Registry<br/>alias champion / challenger)]
    R --> E[Export du champion<br/>+ manifest.json]
    E --> D[Image Docker]
    D --> ECR[(ECR)] --> ECS[ECS Fargate<br/>derrière un ALB]
    ECS --> J[Journal des prédictions<br/>stdout -> CloudWatch]
    ECS --> PM[/metrics Prometheus/]
    J --> DR{finml drift<br/>KS + PSI}
    DR -- dérive --> A[Alerte, réentraînement]
```

Chaîne de déploiement (`.github/workflows/deploy.yml`), à chaque push sur `main` :

1. lint et tests (`ruff`, `pytest`) ;
2. génération des données, entraînement, porte qualité, export des champions ;
3. authentification AWS par OIDC, sans clé stockée dans GitHub ;
4. build de l'image, tag = SHA du commit (tags immuables dans ECR) ;
5. nouvelle révision de la définition de tâche, déploiement ECS, attente de la stabilité.
   Si les health checks échouent, le circuit breaker ECS revient à la version précédente.

## Démarrage rapide

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

finml list                  # cas d'usage déclarés
finml run-all               # données -> entraînement -> Registry -> export dans models/
pytest                      # 21 tests, environ 50 s

uvicorn finml.api:app --port 8000   # http://localhost:8000/docs
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
```

Exemple d'appel :

```bash
curl -s localhost:8000/predict/imputation_comptable -H 'content-type: application/json' \
  -d '{"records":[{"libelle":"CB HOTEL DU PARC NUITEE HOTEL MARS"}]}'
```

```json
{"usecase": "imputation_comptable", "modele_version": 1, "latence_ms": 9.1,
 "predictions": [{"compte": "6251", "intitule": "Voyages et déplacements", "confiance": 1.0,
                  "a_revoir": false, "top3": [...],
                  "exemples_proches": [{"libelle": "FAC N° Hotel du Parc nuitee hotel MARS",
                                        "compte": "6251", "similarite": 0.946}, ...]}]}
```

`a_revoir` passe à `true` sous le seuil de confiance déclaré (`decision_threshold`) :
la proposition part alors en revue humaine.

Prédiction par lots, avec le même contrat de données que l'API :

```bash
finml batch retard_paiement factures.csv predictions.csv
```

Avec Docker (serveur MLflow, pgvector et API) : voir l'en-tête de `docker-compose.yml`.

## Ajouter un cas d'usage

1. Créer `usecases/<nom>/config.yaml` : cible, variables, contrat de données, métrique, seuil minimal, candidats.
2. Fournir les données : un `generate.py` exposant `generate(n_rows, seed)`, ou un CSV à l'emplacement `data.path`.
3. `finml run-all`. L'API expose automatiquement `POST /predict/<nom>`.

Aucune ligne du moteur ne change. Les cas candidats suivants s'ajoutent de cette façon : détection de
fraude sur transactions, défaut de crédit, prévision de trésorerie.

## Ce que le moteur garantit

| Garantie | Mécanisme |
|---|---|
| Données conformes | Contrat déclaré dans la configuration, vérifié à l'entraînement **et** à chaque appel API (422 sinon) |
| Traçabilité | Empreinte SHA-256 des données, SHA Git, paramètres et métriques de chaque candidat dans MLflow |
| Pas de régression | Un nouveau modèle ne devient `champion` que s'il bat l'actuel, sinon il reste `challenger` |
| Pas de modèle faible en production | Porte qualité `min_score` : la CI échoue, rien n'est déployé |
| Pas de trafic vers un conteneur inutilisable | `/health` renvoie 503 tant qu'un modèle attendu n'est pas chargé |
| Image reproductible | Modèles champions embarqués, version servie lisible dans `manifest.json` et `/usecases` |
| Aucun pickle sur le chemin de service | Estimateurs sérialisés avec skops, rechargés seulement si chaque type figure dans une liste blanche. Enveloppe MLflow chargée depuis son code source (« models from code », MLflow 3) |
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

## Limites connues et suites

- Données fictives : les scores ne valent pas performance métier.
- Le Registry MLflow vit dans la CI (SQLite archivé comme artefact). Étape suivante : un serveur MLflow partagé, adossé à RDS et S3.
- Image de 1,5 Go, dont l'essentiel vient de MLflow. Piste : `mlflow-skinny` pour le service.
- La détection de dérive tourne à la demande. Étape suivante : un job planifié (EventBridge) qui exporte le journal CloudWatch, lance `finml drift` et déclenche le pipeline d'entraînement sur alerte.
- La dérive de concept n'est suivie qu'à travers les sorties du modèle. Le vrai signal arrive avec les étiquettes réelles (date de paiement effective, compte corrigé par le comptable), à rapprocher des prédictions journalisées.

## Structure

```
src/finml/         moteur : config, data, models, vector, train, export, api, monitoring, cli
usecases/          un dossier par cas d'usage : config.yaml + generate.py
tests/             tests unitaires et bout en bout (génération -> API)
infra/terraform/   infrastructure AWS (ECS, rolling ou blue/green CodeDeploy)
deploy/k8s/        manifestes Kubernetes (Kustomize, base + overlay prod)
deploy/monitoring/ Prometheus, alertes et leurs tests, Grafana
.github/workflows  CI (lint, tests, Terraform, Kubernetes, alertes), déploiement ECS, déploiement EKS
```

## Stack

Python 3.12, scikit-learn, XGBoost, MLflow 3 (tracking, Registry, alias, models from code), skops,
FastAPI, Prometheus, Grafana, SciPy (KS), pgvector, Docker, Kubernetes (Kustomize), GitHub Actions (OIDC),
Terraform, AWS (ECS Fargate, CodeDeploy, ALB, ECR, CloudWatch, IAM, Secrets Manager).

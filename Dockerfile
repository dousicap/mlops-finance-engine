# Image de service : le code, les configurations et les modèles champions exportés.
# Un seul processus uvicorn par conteneur : les métriques Prometheus restent exactes,
# la montée en charge se fait en ajoutant des tâches ECS.
# Les modèles sont produits AVANT le build (`finml run-all`, en CI) puis copiés :
# l'image est autonome, et la version servie est tracée dans models/<cas>/manifest.json.

FROM python:3.12-slim AS builder
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FINML_ROOT=/app \
    FINML_MODELS_DIR=/app/models \
    MLFLOW_TRACKING_URI=sqlite:////tmp/mlflow.db
WORKDIR /app
# libgomp : bibliothèque OpenMP requise par LightGBM à l'import.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /install /usr/local
COPY usecases ./usecases
COPY models ./models
RUN for u in retard_paiement imputation_comptable defaut_credit maintenance_predictive; do \
        test -f models/$u/MLmodel || exit 1; done \
    && useradd --uid 1001 --no-create-home finml \
    && chown -R finml /app
USER finml
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "finml.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

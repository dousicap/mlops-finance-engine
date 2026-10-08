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
COPY --from=builder /install /usr/local
COPY usecases ./usecases
COPY models ./models
RUN test -f models/retard_paiement/MLmodel && test -f models/imputation_comptable/MLmodel \
    && useradd --uid 1001 --no-create-home finml \
    && chown -R finml /app
USER finml
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "finml.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

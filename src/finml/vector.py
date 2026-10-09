"""Embeddings et recherche vectorielle.

Deux embedders interchangeables :
- TfidfEmbedder : n-grammes de caractères, sans téléchargement, déterministe (défaut, CI, tests).
- SentenceTransformerEmbedder : modèle multilingue Hugging Face (meilleur sur le sens, à activer en config).

Deux index interchangeables :
- index en mémoire NumPy (embarqué dans le modèle MLflow) ;
- PgVectorIndex : table PostgreSQL + extension pgvector, pour un référentiel partagé et volumineux.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class Embedder(Protocol):
    def fit(self, texts: list[str]) -> Embedder: ...
    def encode(self, texts: list[str]) -> np.ndarray: ...


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class TfidfEmbedder:
    """TF-IDF sur n-grammes de caractères, réduit en vecteur dense par SVD si `dim` est fourni.

    La réduction est nécessaire pour stocker les vecteurs dans pgvector (index HNSW limité
    à 2 000 dimensions) et réduit la mémoire de l'index.
    """

    def __init__(self, ngram_range: tuple[int, int] = (2, 5), max_features: int = 20000,
                 dim: int | None = None, seed: int = 42):
        self.svd = TruncatedSVD(n_components=dim, random_state=seed) if dim else None
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=tuple(ngram_range),
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
            max_features=max_features,
        )

    def fit(self, texts: list[str]) -> TfidfEmbedder:
        tfidf = self.vectorizer.fit_transform(texts)
        if self.svd is not None:
            self.svd.fit(tfidf)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        tfidf = self.vectorizer.transform(texts)
        dense = self.svd.transform(tfidf) if self.svd is not None else tfidf.toarray()
        return _l2_normalize(dense.astype(np.float32))


class SentenceTransformerEmbedder:
    """Embeddings sémantiques. Le modèle est chargé à la demande et n'est pas sérialisé."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _get(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, texts: list[str]) -> SentenceTransformerEmbedder:
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._get().encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    def __getstate__(self):
        return {"model_name": self.model_name, "_model": None}


def build_embedder(name: str = "tfidf", **params) -> Embedder:
    if name == "tfidf":
        return TfidfEmbedder(**params)
    if name.startswith("sentence-transformers/"):
        return SentenceTransformerEmbedder(name)
    raise ValueError(f"Embedder inconnu : {name}")


class VectorRetrievalClassifier:
    """Classification par plus proches voisins dans un espace d'embeddings.

    Pour un libellé, on retrouve les k libellés déjà imputés les plus proches et on vote,
    pondéré par la similarité cosinus. Le résultat est explicable : on renvoie les exemples
    qui ont justifié la proposition, comme la partie « retrieval » d'un RAG.
    """

    def __init__(self, k: int = 7, embedder: str = "tfidf", embedder_params: dict | None = None):
        self.k = k
        self.embedder_name = embedder
        self.embedder_params = embedder_params or {}
        self.embedder: Embedder | None = None
        self.vectors: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.texts: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, texts, labels) -> VectorRetrievalClassifier:
        texts = [str(t) for t in texts]
        self.embedder = build_embedder(self.embedder_name, **self.embedder_params).fit(texts)
        self.vectors = self.embedder.encode(texts)
        self.labels = np.asarray([str(x) for x in labels])
        self.texts = np.asarray(texts)
        self.classes_ = np.unique(self.labels)
        return self

    def neighbors(self, texts) -> tuple[np.ndarray, np.ndarray]:
        """Indices et similarités des k voisins (index en mémoire)."""
        q = self.embedder.encode([str(t) for t in texts])
        sims = q @ self.vectors.T
        k = min(self.k, sims.shape[1])
        idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        rows = np.arange(len(q))[:, None]
        order = np.argsort(-sims[rows, idx], axis=1)
        idx = idx[rows, order]
        return idx, sims[rows, idx]

    def class_scores(self, texts) -> np.ndarray:
        idx, sims = self.neighbors(texts)
        class_pos = {c: i for i, c in enumerate(self.classes_)}
        scores = np.zeros((len(idx), len(self.classes_)), dtype=np.float32)
        for r in range(len(idx)):
            for j, s in zip(idx[r], sims[r]):
                scores[r, class_pos[self.labels[j]]] += max(float(s), 0.0)
        totals = scores.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        return scores / totals

    def predict(self, texts) -> np.ndarray:
        return self.classes_[self.class_scores(texts).argmax(axis=1)]

    def explain(self, texts, n: int = 3) -> list[list[dict]]:
        idx, sims = self.neighbors(texts)
        out = []
        for r in range(len(idx)):
            out.append(
                [
                    {"texte": str(self.texts[j]), "classe": str(self.labels[j]), "similarite": round(float(s), 3)}
                    for j, s in list(zip(idx[r], sims[r]))[:n]
                ]
            )
        return out


class PgVectorIndex:
    """Index persistant dans PostgreSQL/pgvector. Utilisé quand FINML_PGVECTOR_DSN est défini."""

    def __init__(self, dsn: str, table: str, dim: int):
        self.dsn, self.table, self.dim = dsn, table, dim

    def _connect(self):
        import psycopg
        from pgvector.psycopg import register_vector

        conn = psycopg.connect(self.dsn, autocommit=True)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        return conn

    def rebuild(self, texts: list[str], labels: list[str], vectors: np.ndarray) -> int:
        with self._connect() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {self.table}")
            conn.execute(
                f"CREATE TABLE {self.table} (id serial PRIMARY KEY, texte text, classe text, "
                f"embedding vector({self.dim}))"
            )
            with conn.cursor() as cur:
                cur.executemany(
                    f"INSERT INTO {self.table} (texte, classe, embedding) VALUES (%s, %s, %s)",
                    [(t, lab, v) for t, lab, v in zip(texts, labels, vectors)],
                )
            conn.execute(
                f"CREATE INDEX ON {self.table} USING hnsw (embedding vector_cosine_ops)"
            )
        return len(texts)

    def search(self, vector: np.ndarray, k: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT texte, classe, 1 - (embedding <=> %s) AS sim FROM {self.table} "
                f"ORDER BY embedding <=> %s LIMIT %s",
                (vector, vector, k),
            ).fetchall()
        return [{"texte": r[0], "classe": r[1], "similarite": round(float(r[2]), 3)} for r in rows]

    def vote(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        scores: dict[str, float] = defaultdict(float)
        for hit in self.search(vector, k):
            scores[hit["classe"]] += max(hit["similarite"], 0.0)
        total = sum(scores.values()) or 1.0
        return sorted(((c, s / total) for c, s in scores.items()), key=lambda x: -x[1])

"""Ligne de commande : finml list | generate | train | export | run-all | index-pgvector | batch | simulate | drift."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC
from pathlib import Path

from finml.config import list_usecases, load_config


def _targets(name: str) -> list[str]:
    return list_usecases() if name == "all" else [name]


def cmd_list(_):
    for name in list_usecases():
        cfg = load_config(name)
        print(f"{name:<24} {cfg.task:<24} {cfg.metric:<14} {cfg.description}")


def cmd_generate(args):
    from finml.data import generate, load, validate

    for name in _targets(args.usecase):
        cfg = load_config(name)
        path = generate(cfg)
        report = validate(load(cfg), cfg)
        print(f"{name} : {report.n_rows} lignes -> {path} | qualité {'OK' if report.ok else report.errors}")
        for w in report.warnings:
            print(f"  avertissement : {w}")


def cmd_train(args):
    from finml.train import QualityGateError, train

    failed = False
    for name in _targets(args.usecase):
        try:
            r = train(name)
        except QualityGateError as e:
            print(f"ÉCHEC porte qualité - {e}", file=sys.stderr)
            failed = True
            continue
        detail = ", ".join(f"{k}={v:.3f}" for k, v in r.all_scores.items())
        print(f"{name} : meilleur={r.best_candidate} {r.metric}={r.score:.3f} -> v{r.version} @{r.alias} [{detail}]")
    if failed:
        sys.exit(1)


def cmd_export(args):
    from finml.export import export_champion

    for name in _targets(args.usecase):
        print(f"{name} : exporté vers {export_champion(name, Path(args.out))}")


def cmd_index_pgvector(args):
    """Charge le référentiel de libellés déjà imputés dans pgvector (cas d'usage texte).

    Le référentiel est une donnée, pas un modèle : il est indexé indépendamment du champion,
    et sert à retrouver les écritures passées les plus proches d'un nouveau libellé.
    """
    from finml.data import load_validated
    from finml.vector import PgVectorIndex, VectorRetrievalClassifier

    cfg = load_config(args.usecase)
    if cfg.task != "text_classification":
        sys.exit("Indexation vectorielle réservée aux cas d'usage texte.")
    df = load_validated(cfg)
    est = VectorRetrievalClassifier(k=args.k, embedder_params={"dim": args.dim}).fit(df[cfg.features.text], df[cfg.target])
    dsn = args.dsn or os.environ["FINML_PGVECTOR_DSN"]
    index = PgVectorIndex(dsn, table=f"ref_{cfg.name}", dim=est.vectors.shape[1])
    n = index.rebuild(est.texts.tolist(), est.labels.tolist(), est.vectors)
    print(f"{n} vecteurs indexés dans {index.table}")
    if args.query:
        vec = est.embedder.encode([args.query])[0]
        for hit in index.search(vec, 3):
            print("  ", hit)


def cmd_simulate(args):
    """Envoie du trafic synthétique à l'API (avec ou sans dérive)."""
    from finml.monitoring import send_traffic, simulate_records

    cfg = load_config(args.usecase)
    records = simulate_records(cfg, args.n, drift=args.drift, seed=args.seed)
    n = send_traffic(args.url, args.usecase, records)
    print(f"{n} prédictions envoyées à {args.url} ({'avec' if args.drift else 'sans'} dérive)")


def cmd_drift(args):
    """Compare le journal de production à la référence du modèle servi. Code retour 3 si dérive."""
    import json
    import time

    import pandas as pd

    from finml.monitoring import PredictionLogger, detect_drift, load_production_log

    cfg = load_config(args.usecase)
    reference = pd.read_csv(Path(args.models_dir) / args.usecase / "artifacts" / "reference.csv")
    prod_path = Path(args.production) if args.production else PredictionLogger(args.log_dir).path(args.usecase)
    production = load_production_log(prod_path)
    report = detect_drift(cfg, reference, production, alpha=args.alpha, ks_min=args.ks_min, psi_max=args.psi_max)

    out = Path(args.out) / f"drift_{args.usecase}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    _log_drift_to_mlflow(cfg.name, report, out)

    print(f"{args.usecase} : {report['statut']} ({report['lignes_production']} lignes) -> {out}")
    for col, v in report["variables"].items():
        mesure = f"ks={v['ks']:.3f} p={v['p_value']:.1e}" if "ks" in v else f"psi={v['psi']:.3f}"
        print(f"  {'DÉRIVE' if v['derive'] else 'ok    '} {col:<28} {mesure}")
    if report["derive"]:
        sys.exit(3)


def _log_drift_to_mlflow(usecase: str, report: dict, path: Path):
    """Historise chaque contrôle de dérive dans MLflow, à côté des entraînements."""
    import mlflow

    from finml.train import default_tracking_uri

    mlflow.set_tracking_uri(default_tracking_uri())
    mlflow.set_experiment(f"finml-{usecase}-monitoring")
    with mlflow.start_run(run_name=f"derive-{report['statut']}"):
        mlflow.set_tags({"usecase": usecase, "statut": report["statut"]})
        mlflow.log_metrics({"lignes_production": report["lignes_production"],
                            "variables_en_derive": len(report.get("variables_en_derive", []))})
        for col, v in report["variables"].items():
            mlflow.log_metric(f"{'ks' if 'ks' in v else 'psi'}_{col}", v.get("ks", v.get("psi")))
        mlflow.log_artifact(str(path))


def cmd_batch(args):
    """Prédiction par lots : CSV en entrée, CSV en sortie, même contrat de données que l'API."""
    from datetime import datetime

    import mlflow
    import pandas as pd

    from finml.data import validate
    from finml.train import default_tracking_uri

    cfg = load_config(args.usecase)
    df = pd.read_csv(args.input)
    report = validate(df, cfg, with_target=False)
    if not report.ok:
        sys.exit(f"Données refusées : {report.errors}")
    local = Path(args.models_dir) / args.usecase
    if (local / "MLmodel").exists():
        uri = str(local)
    else:
        mlflow.set_tracking_uri(default_tracking_uri())
        uri = f"models:/{cfg.registered_model}@champion"
    model = mlflow.pyfunc.load_model(uri)
    X = df[cfg.features.all].astype({c: "float64" for c in cfg.features.numeric})
    out = pd.concat([df.reset_index(drop=True), model.predict(X).reset_index(drop=True)], axis=1)
    out["modele"] = uri
    out["predit_le"] = datetime.now(UTC).isoformat()
    out.to_csv(args.output, index=False)
    print(f"{len(out)} lignes prédites -> {args.output}")


def cmd_run_all(args):
    args.usecase = "all"
    cmd_generate(args)
    cmd_train(args)
    cmd_export(args)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="finml", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(func=cmd_list)
    for cmd, func in [("generate", cmd_generate), ("train", cmd_train)]:
        p = sub.add_parser(cmd)
        p.add_argument("usecase", nargs="?", default="all")
        p.set_defaults(func=func)
    p = sub.add_parser("export")
    p.add_argument("usecase", nargs="?", default="all")
    p.add_argument("--out", default="models")
    p.set_defaults(func=cmd_export)
    p = sub.add_parser("run-all")
    p.add_argument("--out", default="models")
    p.set_defaults(func=cmd_run_all)
    p = sub.add_parser("index-pgvector")
    p.add_argument("usecase")
    p.add_argument("--dsn")
    p.add_argument("--query")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--dim", type=int, default=256)
    p.set_defaults(func=cmd_index_pgvector)
    p = sub.add_parser("simulate")
    p.add_argument("usecase")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--drift", action="store_true")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--url", default="http://localhost:8000")
    p.set_defaults(func=cmd_simulate)
    p = sub.add_parser("batch")
    p.add_argument("usecase")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--models-dir", default="models")
    p.set_defaults(func=cmd_batch)
    p = sub.add_parser("drift")
    p.add_argument("usecase")
    p.add_argument("--production", help="journal JSONL (défaut : <log-dir>/<cas>.jsonl)")
    p.add_argument("--log-dir", default=None)
    p.add_argument("--models-dir", default="models")
    p.add_argument("--out", default="reports")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--ks-min", type=float, default=0.1)
    p.add_argument("--psi-max", type=float, default=0.2)
    p.set_defaults(func=cmd_drift)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

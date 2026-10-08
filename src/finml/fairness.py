"""Audit d'équité d'une décision binaire, mesuré sur le jeu de test au seuil servi.

Pour chaque colonne sensible et chaque groupe :
- taux réel : part de cas positifs observés (ex. défauts) ;
- taux signalé : part de cas que la décision signale (ex. limite réduite) ;
- rappel : part des cas positifs que la décision détecte (égalité des chances) ;
- précision : part des signalements justifiés.

Deux indicateurs par colonne :
- écart de rappel entre groupes (max - min) : un groupe dont les défauts passent inaperçus
  est mal protégé, et le prêteur aussi ;
- ratio de taux favorable (min / max de la part NON signalée) : règle des quatre cinquièmes,
  utilisée en pratique pour repérer un effet disproportionné.

Un groupe trop petit (effectif ou cas positifs sous les minimums) est décrit mais exclu des
comparaisons : un rappel calculé sur une douzaine de cas varie surtout au hasard.

Une alerte ne prouve pas une discrimination et son absence ne prouve pas l'équité : elle
déclenche une revue humaine documentée, comme l'attend l'AI Act pour le scoring de crédit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finml.config import FairnessConfig


def _group_table(y: np.ndarray, flag: np.ndarray, groups: pd.Series) -> dict[str, dict[str, float]]:
    table = {}
    for g in sorted(groups.unique(), key=str):
        m = (groups == g).to_numpy()
        yg, fg = y[m], flag[m]
        pos = yg == 1
        table[str(g)] = {
            "effectif": int(m.sum()),
            "taux_reel": float(pos.mean()),
            "taux_signale": float(fg.mean()),
            "rappel": float(fg[pos].mean()) if pos.any() else float("nan"),
            "precision": float(yg[fg].mean()) if fg.any() else float("nan"),
        }
    return table


def fairness_report(y, flag, sensitive: pd.DataFrame, cfg: FairnessConfig) -> tuple[dict[str, float], dict]:
    """Renvoie (métriques à tracer, détail par groupe). Statut : « ok » ou « alerte »."""
    y, flag = np.asarray(y).astype(int), np.asarray(flag).astype(bool)
    metrics, detail, alerts = {}, {}, []
    for col in cfg.sensitive:
        table = _group_table(y, flag, sensitive[col].reset_index(drop=True))
        sized = {g: t for g, t in table.items() if t["effectif"] >= cfg.min_group_size}
        recalls = [t["rappel"] for t in sized.values()
                   if round(t["taux_reel"] * t["effectif"]) >= cfg.min_group_positives]
        favorable = [1 - t["taux_signale"] for t in sized.values()]
        for g, t in table.items():
            t["compare"] = g in sized
        gap = max(recalls) - min(recalls) if recalls else 0.0
        ratio = min(favorable) / max(favorable) if favorable and max(favorable) > 0 else 1.0
        metrics[f"equite_{col}_ecart_rappel"] = gap
        metrics[f"equite_{col}_ratio_taux_favorable"] = ratio
        if gap > cfg.max_recall_gap:
            alerts.append(f"{col} : écart de rappel {gap:.3f} > {cfg.max_recall_gap}")
        if ratio < cfg.min_favorable_ratio:
            alerts.append(f"{col} : ratio de taux favorable {ratio:.3f} < {cfg.min_favorable_ratio}")
        detail[col] = table
    status = "alerte" if alerts else "ok"
    return metrics, {"statut": status, "alertes": alerts, "groupes": detail}

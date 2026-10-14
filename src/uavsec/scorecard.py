"""How close is a profile to the paper? One number per figure, plus a weighted total.

Reads the latest finished runs of a profile (convergence, beamforming, deployment) and compares
them with the digitised paper values in configs/paper_reference.yaml. Every error is
dimensionless and 0 is a perfect match:

* rate curves (Fig. 4, 5, 6, 9, 10): mean |ours - paper| / max(|paper|, 1)
* Fig. 7: distance to the paper's marker divided by DISTANCE_SCALE_M
* Fig. 8: mean |ours - paper| of the heuristic-to-Graph-DRL rate ratios

MLP curves are reported but NOT scored: a weaker MLP must never improve the score.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .channel import load_paper_reference
from .utils import latest_run, load_json

DISTANCE_SCALE_M = 50.0          # 50 m off the paper's marker counts as error 1
SCORED_METHODS = ("GNN", "OPT")
WEIGHTS = {                      # figure -> weight in the total (rate curves of Fig. 9 weigh most)
    "fig4": 1.0, "fig5": 0.5, "fig6": 0.5, "fig7_optimum": 1.0, "fig7_sac": 1.0,
    "fig8": 1.0, "fig9a": 2.0, "fig9b": 1.0, "fig9c": 1.0, "fig10": 0.5,
}


def relative_error(ours: np.ndarray, paper: np.ndarray) -> float:
    ours, paper = np.asarray(ours, dtype=float), np.asarray(paper, dtype=float)
    return float(np.mean(np.abs(ours - paper) / np.maximum(np.abs(paper), 1.0)))


def _final_loss(df: pd.DataFrame, group: str, last: int = 10) -> pd.Series:
    """Mean training loss over the last ``last`` epochs, averaged over seeds."""
    tail = df[df["epoch"] > df["epoch"].max() - last]
    return tail.groupby(group)["train_loss"].mean()


def _sweep(sweeps: pd.DataFrame, sweep: str, method: str, xs: list[float]) -> np.ndarray:
    sub = sweeps[(sweeps["sweep"] == sweep) & (sweeps["method"] == method)].set_index("value")["rate_mean"]
    # match each paper x to the nearest swept value (noise values are floats in W)
    return np.array([sub.iloc[int(np.argmin(np.abs(np.log(sub.index.to_numpy(float) + 1e-30) - np.log(x + 1e-30))
                                                if sweep == "noise" else np.abs(sub.index.to_numpy(float) - x)))]
                     for x in xs])


def paper_scorecard(output_path: Path) -> dict:
    ref = load_paper_reference()
    runs = {name: latest_run(output_path, name) for name in ("convergence", "beamforming", "deployment")}
    items: dict[str, dict] = {}

    if runs["convergence"]:
        losses = _final_loss(pd.read_csv(runs["convergence"] / "convergence.csv"), "model")
        items["fig4"] = {"error": relative_error([losses["GNN"]], [ref["fig4"]["gnn_loss"]]),
                         "ours": {"GNN": losses["GNN"], "MLP (not scored)": losses["MLP"]},
                         "paper": {"GNN": ref["fig4"]["gnn_loss"], "MLP": ref["fig4"]["mlp_loss"]}}
        tl = _final_loss(pd.read_csv(runs["convergence"] / "transfer.csv"), "method")
        ours5 = [tl["Transfer learning"], tl["Re-training"]]
        paper5 = [ref["fig5"]["transfer_loss"], ref["fig5"]["retrain_loss"]]
        items["fig5"] = {"error": relative_error(ours5, paper5), "ours": ours5, "paper": paper5}

    if runs["beamforming"]:
        sweeps = pd.read_csv(runs["beamforming"] / "sweeps.csv")
        for fig, sweep in (("fig9a", "users"), ("fig9b", "power"), ("fig9c", "noise")):
            xs = ref[fig]["x"]
            ours = {m: _sweep(sweeps, sweep, m, xs) for m in (*SCORED_METHODS, "MLP")}
            errs = {m: relative_error(ours[m], ref[fig][m]) for m in SCORED_METHODS}
            items[fig] = {"error": float(np.mean(list(errs.values()))), "per_method": errs,
                          "mlp_error_not_scored": relative_error(ours["MLP"], ref[fig]["MLP"]),
                          "ours": {m: np.round(v, 2).tolist() for m, v in ours.items()},
                          "paper": {m: ref[fig][m] for m in (*SCORED_METHODS, "MLP")}}
        cdf = pd.read_csv(runs["beamforming"] / "cdf.csv")
        med = cdf.groupby(["num_users", "method"])["rate"].median()
        ours10, paper10 = [], []
        for key, k in (("K8", 8), ("K12", 12)):
            for m in SCORED_METHODS:
                if (k, m) in med.index:
                    ours10.append(med[(k, m)])
                    paper10.append(ref["fig10"][key][m])
        if ours10:
            items["fig10"] = {"error": relative_error(ours10, paper10), "ours": ours10, "paper": paper10}

    if runs["deployment"]:
        dep_dir = runs["deployment"]
        m = load_json(dep_dir / "metrics.json")
        items["fig7_optimum"] = {"error": m["optimum_distance_to_paper_m"] / DISTANCE_SCALE_M,
                                 "ours_m": m["optimum_distance_to_paper_m"]}
        sac_end = np.asarray(m["final_position"])
        sac_dist = float(np.linalg.norm(sac_end - np.asarray(ref["fig7"]["sac_end"])))
        items["fig7_sac"] = {"error": sac_dist / DISTANCE_SCALE_M, "ours_m": sac_dist}
        rates = pd.read_csv(dep_dir / "deployment.csv").set_index("strategy")["rate"]
        ref8 = ref["fig8"]
        ratio = {s: rates[s] / rates["Graph-DRL"] for s in ref8 if s != "Graph-DRL"}
        ratio_paper = {s: ref8[s] / ref8["Graph-DRL"] for s in ratio}
        items["fig8"] = {"error": float(np.mean([abs(ratio[s] - ratio_paper[s]) for s in ratio])),
                         "ours": {s: round(v, 3) for s, v in ratio.items()},
                         "paper": {s: round(v, 3) for s, v in ratio_paper.items()}}
        episodes = pd.read_csv(dep_dir / "episodes.csv").tail(20)
        ours6 = [episodes["episode_reward"].mean(), episodes["mean_secrecy_rate"].mean()]
        items["fig6"] = {"error": relative_error(ours6, [ref["fig6"]["episode_reward"], ref["fig6"]["secrecy_rate"]]),
                         "ours": ours6, "paper": [ref["fig6"]["episode_reward"], ref["fig6"]["secrecy_rate"]]}

    scored = {k: v["error"] for k, v in items.items() if k in WEIGHTS}
    total = sum(WEIGHTS[k] * e for k, e in scored.items()) / sum(WEIGHTS[k] for k in scored) if scored else float("nan")
    return {"total_error": total, "errors": scored, "weights": WEIGHTS, "items": items,
            "runs": {k: (v.name if v else None) for k, v in runs.items()}}

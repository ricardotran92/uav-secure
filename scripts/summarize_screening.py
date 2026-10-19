"""Summarise a GNN screening round written by scripts/server/screen_gnn.sh.

<root>/<variant>/gnn_optimum_*/metrics.json      K = 8 vs OPT and the Fig. 7 optimum per seed
<root>/<variant>/eval_gnn_seed*/beamforming_*/   Fig. 9a/9b sweeps per checkpoint

Per variant (mean over seeds): GNN / OPT at K = 8, GNN at K = 1 and 40 dBm, the Fig. 9a/9b
GNN-curve error (scorecard formula, GNN only) and the median Fig. 7 distance. Eval-only
variants (no gnn_optimum run) get the sweep metrics only.

  python scripts/summarize_screening.py outputs/studies_r14 [--json rows.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from uavsec.channel import load_paper_reference
from uavsec.scorecard import relative_error


def sweep_metrics(csv: Path, ref: dict) -> dict:
    d = pd.read_csv(csv)
    rate = d.pivot_table(index=["sweep", "value"], columns="method", values="rate_mean")
    out = {}
    for fig, sweep in (("fig9a", "users"), ("fig9b", "power")):
        gnn = rate.loc[sweep, "GNN"]
        out[f"{fig}_err"] = relative_error([gnn.loc[float(x)] for x in ref[fig]["x"]], ref[fig]["GNN"])
    out["gnn_k8_over_opt"] = rate.loc[("users", 8.0), "GNN"] / rate.loc[("users", 8.0), "OPT"]
    out["gnn_k8"] = rate.loc[("users", 8.0), "GNN"]
    out["gnn_k1"] = rate.loc[("users", 1.0), "GNN"]
    out["gnn_40dbm"] = rate.loc[("power", 40.0), "GNN"]
    return out


def summarise(root: Path) -> pd.DataFrame:
    ref = load_paper_reference()
    rows = []
    for var in sorted(p for p in root.iterdir() if p.is_dir()):
        runs = sorted(var.glob("gnn_optimum_*/metrics.json"))
        fig7 = {s["seed"]: s for s in json.loads(runs[-1].read_text())["seeds"]} if runs else {}
        for ev in sorted(var.glob("eval_gnn_seed*")):
            csvs = sorted(ev.glob("beamforming_*/sweeps.csv"))
            if not csvs:
                continue
            seed = int(ev.name.removeprefix("eval_gnn_seed"))
            row = {"variant": var.name, "seed": seed, **sweep_metrics(csvs[-1], ref)}
            if seed in fig7:
                row["val_rate"] = fig7[seed]["best_val_rate"]
                row["fig7_distance_m"] = fig7[seed]["distance_to_paper_opt_m"]
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("--json", type=Path, help="write the per-variant means here")
    args = parser.parse_args()
    seeds = summarise(args.root)
    agg = {c: "mean" for c in seeds.columns if c not in ("variant", "seed")}
    if "fig7_distance_m" in agg:
        agg["fig7_distance_m"] = "median"
    table = seeds.groupby("variant").agg({**agg, "seed": "count"}).rename(columns={"seed": "seeds"})
    table["fig9ab_err"] = table["fig9a_err"] + table["fig9b_err"]
    table = table.sort_values("fig9ab_err")
    pd.set_option("display.width", 200)
    print(seeds.round(3).to_string(index=False), end="\n\n")
    print(table.round(3).to_string())
    if args.json:
        records = [{"variant": v, **{k: (None if pd.isna(x) else round(float(x), 3)) for k, x in r.items()}}
                   for v, r in table.iterrows()]
        args.json.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

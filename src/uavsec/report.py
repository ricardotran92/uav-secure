"""Build ``outputs/report/report.md`` from the latest finished run of each experiment.

Figures are regenerated from the saved CSV/NPZ/JSON artefacts only (no model is
re-run), which also proves every number in the report is traceable to a run.
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import plotting
from .config import Config
from .evaluation.beamforming import PAPER_METHODS
from .utils import latest_run, load_json, save_json

EXPERIMENTS = ("convergence", "beamforming", "deployment", "deployment_sensitivity")

# Table I of the paper vs. this implementation (keys into Config).
TABLE_I = [
    ("Area size", "200 m x 200 m", lambda c: f"{c.system.area_size:g} m x {c.system.area_size:g} m"),
    ("No. of users", "8", lambda c: c.system.num_users),
    ("No. of antennas", "8", lambda c: c.system.num_antennas),
    ("UAV altitude", "100 m", lambda c: f"{c.system.uav_altitude:g} m"),
    ("Rician factor", "10 dB", lambda c: f"{c.system.rician_factor_db:g} dB"),
    ("Power budget", "1 W", lambda c: f"{c.system.power_max_w:g} W"),
    ("Path loss (LoS)", "30 + 22 log(d) dB",
     lambda c: f"{c.system.path_loss_a_db:g} + {c.system.path_loss_b:g} {c.system.path_loss_log}(d) dB, "
               f"d = {'3-D' if c.system.distance_model == '3d' else 'horizontal'} distance"),
    ("Noise power", "1.2e-13 W", lambda c: f"{c.system.noise_power_w:.2g} W"),
    ("GNN layers / LR / epochs / batch", "5 / 0.005 / 300 / 512",
     lambda c: f"{c.gnn.num_layers} / {c.gnn.lr:g} / {c.gnn.epochs} / {c.gnn.batch_size}"),
    ("SAC LR / episodes / mini-batch / buffer / discount", "3e-4 / 500 / 64 / 1e6 / 0.99",
     lambda c: f"{c.sac.lr:g} / {c.sac.episodes} / {c.sac.batch_size} / {c.sac.buffer_size:.0e} / {c.sac.gamma:g}"),
]

ASSUMPTIONS = [
    "UAV array: half-wavelength ULA along the {axis}-axis; LoS component = array steering vector.",
    "Eavesdropper k is placed exactly `eve_distance` from user k at a uniformly random bearing (clipped to the area).",
    "GNN: weights shared by all nodes, hidden width {hidden}, max aggregation, PReLU, e_k^(0) = 0; "
    "inputs are channels scaled by sqrt(P / SNR_ref) for numerical conditioning.",
    "Eq. (19) is applied with sqrt(P^max) so that the power constraint (11) holds with equality.",
    "GNN/MLP training: {batches} fresh mini-batches per epoch (random topology, UAV position and fading); "
    "gradient-norm clipping at {clip}; the checkpoint with the best validation rate is kept.",
    "MLP: hidden layers {mlp}, LR {mlp_lr}; one MLP per K, since its input size depends on K.",
    "OPT: the scheme of [45] is not specified in the paper; it is replaced by projected-gradient (Adam) ascent "
    "on the sum secrecy rate with {restarts} restarts (MRT, ZF, random), {iters} iterations each.",
    "SAC: Stable-Baselines3 implementation; {T} steps per episode, max move {step} m per step, "
    "networks {arch}, fixed user/eavesdropper layout (seed {topo}), fresh fading every step.",
    "Deployment strategies are scored by the expected sum secrecy rate (mean over fading draws) with GNN beamforming.",
]


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table (no extra dependency)."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "-" if np.isnan(v) else format(v, floatfmt)
        return str(v)
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def pm(stat: dict) -> str:
    return f"{stat['mean']:.3f} ± {stat['std']:.3f}"


class ReportBuilder:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.out = cfg.output_path.parent / "report" if cfg.output_path.name == "runs" else cfg.output_path / "report"
        self.figs = self.out / "figs"
        self.runs = {name: latest_run(cfg.output_path, name) for name in EXPERIMENTS}
        self.metrics = {n: load_json(d / "metrics.json") for n, d in self.runs.items() if d}
        self.sections: list[str] = []

    def figure(self, fig, name: str, caption: str) -> str:
        plotting.save(fig, self.figs / name)
        return f"![{caption}](figs/{name}.png)\n\n*{caption}*"

    # ---------------------------------------------------------------- sections
    def setup(self) -> None:
        c = self.cfg
        table = pd.DataFrame([(n, paper, ours(c)) for n, paper, ours in TABLE_I],
                             columns=["Parameter", "Paper (Table I)", "This run"])
        assumptions = [a.format(axis=c.system.array_axis, hidden=c.gnn.hidden_dim, batches=c.gnn.batches_per_epoch, mlp=c.mlp.hidden_dims,
                                mlp_lr=c.mlp.lr, clip=c.gnn.grad_clip, restarts=c.opt.restarts, iters=c.opt.iters,
                                T=c.sac.episode_length, step=c.sac.max_step_m, arch=c.sac.net_arch,
                                topo=c.sac.topology_seed) for a in ASSUMPTIONS]
        self.sections.append("## 1. Simulation setup\n\n" + md_table(table) +
                             "\n\n**Assumptions where the paper is silent**\n\n" +
                             "\n".join(f"- {a}" for a in assumptions))

    def convergence(self) -> None:
        run, m = self.runs["convergence"], self.metrics["convergence"]
        conv = pd.read_csv(run / "convergence.csv")
        transfer = pd.read_csv(run / "transfer.csv")
        bf = pd.DataFrame([{"Model": name, "Final loss": pm(s["final_loss"]),
                            "Best val. sum secrecy rate (bps/Hz)": pm(s["best_val_sum_secrecy_rate"]),
                            "Train time (s)": pm(s["train_time_s"]), "Parameters": s["num_parameters"]}
                           for name, s in m["beamforming"].items()])
        tr = pd.DataFrame([{"Method": name, "Initial val. rate": pm(s["initial_val_rate"]),
                            "Best val. rate (bps/Hz)": pm(s["best_val_sum_secrecy_rate"]),
                            "Epochs to 95% of re-training": s["epochs_to_95pct_of_retrain"]}
                           for name, s in m["transfer"].items()])
        self.sections.append(
            "## 2. GNN security beamforming (Fig. 4, Fig. 5)\n\n" +
            self.figure(plotting.fig4_convergence(conv), "fig4_convergence",
                        f"Fig. 4 - training loss, mean and min-max band over {conv['seed'].nunique()} seeds") +
            "\n\n" + md_table(bf) + "\n\n" +
            self.figure(plotting.fig5_transfer(transfer), "fig5_transfer",
                        f"Fig. 5 - adapting the K={m['num_users']} GNN to K={m['transfer_users']} users") +
            "\n\n" + md_table(tr))

    def beamforming(self) -> None:
        run = self.runs["beamforming"]
        sweeps = pd.read_csv(run / "sweeps.csv")
        cdf = pd.read_csv(run / "cdf.csv")
        timing = pd.read_csv(run / "timing.csv")
        plotted = sweeps[sweeps["method"].isin(PAPER_METHODS)]
        parts = ["## 3. Beamforming benchmarks (Fig. 9, Fig. 10, Fig. 11)\n",
                 self.figure(plotting.fig9_sweeps({s: d for s, d in plotted.groupby("sweep")}), "fig9_sweeps",
                             "Fig. 9 - mean sum secrecy rate (error bars: std over test instances)")]
        labels = {"users": "K", "power": "P (dBm)", "noise": "noise (W)"}
        for sweep, df in sweeps.groupby("sweep", sort=False):
            pivot = df.pivot_table(index="value", columns="method", values="rate_mean")
            pivot = pivot[[c for c in (*PAPER_METHODS, "ZF", "MRT") if c in pivot]]
            pivot["GNN/OPT"] = pivot["GNN"] / pivot["OPT"]
            pivot = pivot.reset_index()
            pivot["value"] = pivot["value"].map(lambda v: f"{v:.2e}" if sweep == "noise" else f"{v:g}")
            pivot = pivot.rename(columns={"value": labels.get(sweep, sweep)})
            parts.append(f"**Sweep: {sweep}** (sum secrecy rate, bps/Hz; ZF/MRT are extra closed-form references)\n\n"
                         + md_table(pivot, ".2f"))
        stats = cdf.groupby(["num_users", "method"])["rate"].describe(percentiles=[0.1, 0.5, 0.9])
        stats = stats[["mean", "10%", "50%", "90%"]].reset_index()
        parts.append(self.figure(plotting.fig10_cdf(cdf[cdf["method"].isin(PAPER_METHODS)]), "fig10_cdf",
                                 "Fig. 10 - CDF of the sum secrecy rate (GNN trained only at the default K)"))
        parts.append(md_table(stats, ".2f"))
        per_run = timing.pivot_table(index="num_users", columns="method", values="seconds_per_run").reset_index()
        parts.append(self.figure(plotting.fig11_timing(timing), "fig11_timing",
                                 "Fig. 11 - total time: networks over many inferences vs one OPT solve"))
        parts.append("Seconds per single instance:\n\n" + md_table(per_run, ".2e"))
        self.sections.append("\n\n".join(parts))

    def deployment(self) -> None:
        run, m = self.runs["deployment"], self.metrics["deployment"]
        episodes = pd.read_csv(run / "episodes.csv")
        traj = pd.read_csv(run / "trajectory.csv")
        dep = pd.read_csv(run / "deployment.csv")
        scen = load_json(run / "scenario.json")
        users, eves = np.asarray(scen["users"]), np.asarray(scen["eves"])
        rate_map = np.load(run / "rate_map.npz")
        area = self.cfg.system.area_size
        headline = pd.DataFrame([
            ("SAC final position (m)", "({:.1f}, {:.1f})".format(*m["final_position"])),
            ("Exhaustive-search optimum (m)", "({:.1f}, {:.1f})".format(*m["optimum_position"])),
            ("Distance to optimum (m)", f"{m['distance_to_optimum_m']:.1f}"),
            ("Graph-DRL rate / optimum rate", f"{m['graph_drl_vs_optimum_pct']:.1f} %"),
            ("Gain over best heuristic", f"{m['graph_drl_gain_over_best_heuristic_pct']:+.1f} %"),
            ("SAC convergence episode (MA reaches 95% of final)", m["sac_convergence_episode"]),
            ("Optimum found with", f"{m['search_beamformer'].upper()} beamforming"),
            ("Optimum for GNN beamforming (m)", "({:.1f}, {:.1f})".format(*m["gnn_surface_optimum_position"])),
        ] + ([
            ("Optimum for OPT beamforming, reference only (m)",
             "({:.1f}, {:.1f})".format(*m["opt_surface_optimum_position"])),
        ] if "opt_surface_optimum_position" in m else []) + ([
            ("Paper optimum, Fig. 7 (m)", "({:.1f}, {:.1f})".format(*m["paper_optimum_position"])),
            ("Our optimum - paper optimum (m)", f"{m['optimum_distance_to_paper_m']:.1f}"),
            ("SAC final position - paper optimum (m)", f"{m['sac_final_distance_to_paper_optimum_m']:.1f}"),
        ] if "paper_optimum_position" in m else []), columns=["Metric", "Value"])
        paper_opt = np.asarray(m["paper_optimum_position"]) if "paper_optimum_position" in m else None
        seed_table = ""
        if (run / "seed_optima.csv").exists():
            seeds = pd.read_csv(run / "seed_optima.csv")
            seed_table = ("**GNN-based optimum for every Phase-2 training seed** (the deployed GNN is the seed "
                          "with the best validation rate, chosen before any deployment result is seen):\n\n"
                          + md_table(seeds, ".1f") + "\n\n")
        self.sections.append(
            "## 4. SAC-based UAV deployment (Fig. 6, Fig. 7, Fig. 8)\n\n" +
            self.figure(plotting.fig6_sac(episodes), "fig6_sac", "Fig. 6 - SAC training convergence") + "\n\n" +
            self.figure(plotting.fig7_trajectory(users, eves, traj[["x", "y"]].to_numpy(), np.asarray(m["optimum_position"]),
                                                 area, (rate_map["grid_x"], rate_map["grid_y"], rate_map["rate"]),
                                                 paper_optimum=paper_opt),
                        "fig7_trajectory", "Fig. 7 - learned trajectory over the expected-secrecy-rate map") + "\n\n" +
            md_table(headline) + "\n\n" + seed_table +
            self.figure(plotting.fig8_deployment(dep, users, eves, area), "fig8_deployment",
                        "Fig. 8 - expected sum secrecy rate per deployment strategy") + "\n\n" +
            md_table(dep.rename(columns={"rate": "rate (bps/Hz)"}), ".2f"))

    def sensitivity(self) -> None:
        run, m = self.runs["deployment_sensitivity"], self.metrics["deployment_sensitivity"]
        table = pd.read_csv(run / "sensitivity.csv")
        table["paper opt, % of max"] = 100 * table["paper_opt_relative_rate"]
        table = table[["variant", "opt_x", "opt_y", "distance_to_paper_opt_m", "optimum_rate",
                       "paper opt, % of max", "fig8_profile_error", "ratio[Area center]"]].rename(columns={
            "opt_x": "opt x", "opt_y": "opt y", "distance_to_paper_opt_m": "to paper opt (m)",
            "optimum_rate": "rate at opt", "fig8_profile_error": "Fig. 8 profile error",
            "ratio[Area center]": "area center / opt"})
        shutil.copyfile(run / "figs" / "sensitivity_maps.png", self.figs / "sensitivity_maps.png")
        profile = ", ".join(f"{k} {v:.2f}" for k, v in m["paper_fig8_profile"].items())
        intro = (f"Beamformer: {m['beamformer'].upper()}. Paper optimum: "
                 f"({m['paper_optimum'][0]:.0f}, {m['paper_optimum'][1]:.0f}); paper Fig. 8 heuristic/optimum "
                 f"ratios: {profile}. The Fig. 8 profile error is the mean absolute difference of these ratios.")
        self.sections.append(
            "## 5. Which channel interpretation reproduces the paper? (sensitivity study)\n\n" + intro + "\n\n" +
            md_table(table, ".2f") + "\n\n![Rate maps per interpretation](figs/sensitivity_maps.png)\n\n"
            "*Relative expected secrecy-rate maps; star = our optimum, X = paper optimum.*")

    def provenance(self) -> None:
        rows = [(name, self.runs[name].name if self.runs[name] else "missing",
                 self.metrics.get(name, {}).get("git", "-"),
                 f"{self.metrics[name]['wall_time_s'] / 60:.1f}" if name in self.metrics else "-")
                for name in EXPERIMENTS]
        self.sections.append("## 6. Provenance\n\n" + md_table(pd.DataFrame(
            rows, columns=["Experiment", "Run directory", "Git commit", "Wall time (min)"])) +
            "\n\nRegenerate: `python scripts/make_report.py` (figures are rebuilt from the CSV files above).")

    def build(self) -> Path:
        if self.figs.exists():
            shutil.rmtree(self.figs)
        self.figs.mkdir(parents=True)
        self.setup()
        for name, section in (("convergence", self.convergence), ("beamforming", self.beamforming),
                              ("deployment", self.deployment), ("deployment_sensitivity", self.sensitivity)):
            if name in self.metrics:
                section()
            else:
                self.sections.append(f"## {name}\n\n_No finished `{name}` run found in {self.cfg.output_path}._")
        self.provenance()
        header = ("# Reproduction report: Deep Graph RL for UAV-Enabled Multi-User Secure Communications\n\n"
                  f"Tang et al., IEEE TMC 2025. Generated {datetime.now():%Y-%m-%d %H:%M}.")
        report = self.out / "report.md"
        report.write_text("\n\n".join([header, *self.sections]) + "\n", encoding="utf-8")
        save_json(self.metrics, self.out / "summary.json")
        return report


def build_report(cfg: Config) -> Path:
    return ReportBuilder(cfg).build()

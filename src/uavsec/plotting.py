"""Figure functions (one per paper figure). Inputs are the CSV tables saved by the
experiments, so every figure can be regenerated without re-running anything.

Colour follows the method, never its rank: the same method has the same colour
in every figure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Validated categorical palette (fixed order), see the dataviz reference palette.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
METHOD_COLORS = {
    "GNN": PALETTE[0], "Graph-DRL": PALETTE[0],
    "OPT": PALETTE[1], "Exhaustive search": PALETTE[1],
    "MLP": PALETTE[2],
    "ZF": PALETTE[3], "Area center": PALETTE[3],
    "MRT": PALETTE[4], "Geometric center": PALETTE[4],
    "Circumcenter": PALETTE[6], "Polygon centroid": PALETTE[7],
    "Transfer learning": PALETTE[0], "Re-training": PALETTE[7],
}
METHOD_MARKERS = {"GNN": "o", "OPT": "^", "MLP": "s", "ZF": "D", "MRT": "v"}
USER_COLOR, EVE_COLOR, INK, MUTED = "#008300", "#e34948", "#0b0b0b", "#52514e"

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "lines.markersize": 6, "legend.frameon": False,
})


def _color(name: str) -> str:
    return METHOD_COLORS.get(name, MUTED)


def save(fig: plt.Figure, path: Path) -> list[Path]:
    """Save as PNG (slides/report) and PDF (vector, for LaTeX/Word)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    outs = [path.with_suffix(".png"), path.with_suffix(".pdf")]
    for out in outs:
        fig.savefig(out)
    plt.close(fig)
    return outs


def _band_plot(ax: plt.Axes, df: pd.DataFrame, x: str, y: str, group: str) -> None:
    """Mean line with a min-max band over seeds/trials for each group."""
    for name, sub in df.groupby(group, sort=False):
        stats = sub.groupby(x)[y].agg(["mean", "min", "max"])
        ax.fill_between(stats.index, stats["min"], stats["max"], color=_color(name), alpha=0.2, lw=0)
        ax.plot(stats.index, stats["mean"], color=_color(name), label=name)


def fig4_convergence(df: pd.DataFrame) -> plt.Figure:
    """Fig. 4: training loss of GNN vs MLP (columns: model, seed, epoch, train_loss)."""
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    _band_plot(ax, df, "epoch", "train_loss", "model")
    ax.set(xlabel="Epoch", ylabel="Loss  (− sum secrecy rate)", title="GNN vs MLP training convergence (K=8)")
    ax.legend()
    return fig


def fig5_transfer(df: pd.DataFrame) -> plt.Figure:
    """Fig. 5: transfer learning vs re-training for a larger K (columns: method, epoch, train_loss)."""
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    _band_plot(ax, df, "epoch", "train_loss", "method")
    k = int(df["num_users"].iloc[0])
    ax.set(xlabel="Epoch", ylabel="Loss  (− sum secrecy rate)", title=f"Adapting the GNN to K={k} users")
    ax.legend()
    return fig


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, min(window, len(values)))
    return pd.Series(values).rolling(window, min_periods=1).mean().to_numpy()


def fig6_sac(df: pd.DataFrame, window: int = 20) -> plt.Figure:
    """Fig. 6: SAC convergence. Two stacked panels instead of a dual y-axis."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.8, 5.0), sharex=True)
    for ax, col, label in ((ax1, "episode_reward", "Episode reward"),
                           (ax2, "mean_secrecy_rate", "Mean sum secrecy\nrate (bps/Hz)")):
        ax.plot(df["episode"], df[col], color=PALETTE[0], alpha=0.25, lw=1.0)
        ax.plot(df["episode"], _moving_average(df[col].to_numpy(), window), color=PALETTE[0])
        ax.set_ylabel(label)
    ax1.set_title(f"SAC training (thin: per episode, bold: {window}-episode average)")
    ax2.set_xlabel("Episode")
    return fig


def draw_nodes(ax: plt.Axes, users: np.ndarray, eves: np.ndarray) -> None:
    ax.scatter(users[:, 0], users[:, 1], marker="P", s=70, color=USER_COLOR, label="Legitimate user", zorder=3)
    ax.scatter(eves[:, 0], eves[:, 1], marker="o", s=45, color=EVE_COLOR, label="Eavesdropper", zorder=3)


def fig7_trajectory(users: np.ndarray, eves: np.ndarray, traj: np.ndarray, optimum: np.ndarray,
                    area: float, heatmap: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
                    paper_optimum: np.ndarray | None = None) -> plt.Figure:
    """Fig. 7: learned UAV trajectory vs the exhaustive-search optimum (optional rate heatmap)."""
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    if heatmap is not None:
        gx, gy, rate = heatmap
        mesh = ax.pcolormesh(gx, gy, rate, cmap="Blues", shading="auto", alpha=0.85, zorder=0)
        fig.colorbar(mesh, ax=ax, label="Expected sum secrecy rate (bps/Hz)", shrink=0.85)
    draw_nodes(ax, users, eves)
    ax.plot(traj[:, 0], traj[:, 1], ls=":", color=INK, lw=1.6, marker=".", ms=3, label="SAC trajectory", zorder=2)
    ax.scatter(*traj[0], marker="^", s=80, color=PALETTE[6], label="Start", zorder=4)
    ax.scatter(*traj[-1], marker="s", s=70, color=PALETTE[0], label="End", zorder=4)
    ax.scatter(*optimum, marker="*", s=180, color=PALETTE[1], edgecolor="white", label="Optimum (search)", zorder=5)
    if paper_optimum is not None:
        ax.scatter(*paper_optimum, marker="X", s=110, color=PALETTE[7], edgecolor="white",
                   label="Paper optimum (Fig. 7)", zorder=5)
    ax.set(xlim=(0, area), ylim=(0, area), xlabel="x (m)", ylabel="y (m)", aspect="equal",
           title="UAV deployment learned by SAC")
    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)
    return fig


def fig8_deployment(df: pd.DataFrame, users: np.ndarray, eves: np.ndarray, area: float) -> plt.Figure:
    """Fig. 8: secrecy rate of deployment strategies (columns: strategy, x, y, rate, rate_std)."""
    fig, (ax_map, ax_bar) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.2]})
    draw_nodes(ax_map, users, eves)
    for _, row in df.iterrows():
        ax_map.scatter(row["x"], row["y"], marker="X", s=110, color=_color(row["strategy"]),
                       edgecolor="white", linewidth=1.2, label=row["strategy"], zorder=4)
    ax_map.set(xlim=(0, area), ylim=(0, area), aspect="equal", xlabel="x (m)", ylabel="y (m)",
               title="UAV positions per strategy")
    ax_map.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)

    order = df.sort_values("rate", ascending=True)
    colors = [_color(s) for s in order["strategy"]]
    ax_bar.barh(order["strategy"], order["rate"], xerr=order.get("rate_std"), color=colors,
                height=0.6, error_kw={"ecolor": MUTED, "lw": 1})
    for y, v in enumerate(order["rate"]):
        ax_bar.text(v, y, f"  {v:.2f}", va="center", color=INK, fontsize=9)
    ax_bar.set(xlabel="Expected sum secrecy rate (bps/Hz)", title="Achieved secrecy rate")
    ax_bar.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def fig9_sweeps(dfs: dict[str, pd.DataFrame]) -> plt.Figure:
    """Fig. 9a-c: rate vs number of users / transmit power / noise power.

    Each frame has columns: value, method, rate_mean, rate_std.
    """
    specs = {
        "users": ("Number of users K", "(a) vs number of users"),
        "power": ("Transmit power (dBm)", "(b) vs transmit power"),
        "noise": ("Noise power (W, log scale)", "(c) vs noise power"),
    }
    present = [k for k in specs if k in dfs]
    fig, axes = plt.subplots(1, len(present), figsize=(4.4 * len(present), 3.6), squeeze=False)
    for ax, key in zip(axes[0], present):
        df = dfs[key]
        for method, sub in df.groupby("method", sort=False):
            ax.errorbar(sub["value"], sub["rate_mean"], yerr=sub["rate_std"], color=_color(method),
                        marker=METHOD_MARKERS.get(method, "o"), capsize=2, elinewidth=0.8, label=method)
        xlabel, title = specs[key]
        ax.set(xlabel=xlabel, ylabel="Sum secrecy rate (bps/Hz)", title=title)
        if key == "noise":
            ax.set_xscale("log")
    axes[0][0].legend()
    fig.suptitle("GNN trained once at the default setting, applied without retraining", fontsize=10, color=MUTED)
    fig.tight_layout()
    return fig


def fig10_cdf(df: pd.DataFrame) -> plt.Figure:
    """Fig. 10: CDF of sum secrecy rate (columns: method, num_users, rate)."""
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    styles = ["-", "--", ":", "-."]
    for i, (k, sub_k) in enumerate(df.groupby("num_users")):
        for method, sub in sub_k.groupby("method", sort=False):
            x = np.sort(sub["rate"].to_numpy())
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), ls=styles[i % len(styles)],
                    color=_color(method), label=f"{method}, K={k}")
    ax.set(xlabel="Sum secrecy rate (bps/Hz)", ylabel="Cumulative probability", title="CDF of achieved secrecy rate")
    ax.legend(fontsize=8)
    return fig


def fig11_timing(df: pd.DataFrame) -> plt.Figure:
    """Fig. 11: run time (columns: method, num_users, seconds, runs). Log scale: values span decades."""
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    methods = list(dict.fromkeys(df["method"]))
    ks = sorted(df["num_users"].unique())
    width = 0.8 / len(methods)
    x = np.arange(len(ks))
    for i, method in enumerate(methods):
        sub = df[df["method"] == method].set_index("num_users").loc[ks]
        runs = int(sub["runs"].iloc[0])
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, sub["seconds"], width * 0.92,
               color=_color(method), label=f"{method} ({runs} run{'s' if runs > 1 else ''})")
    ax.set_xticks(x, [str(k) for k in ks])
    ax.set_yscale("log")
    ax.set(xlabel="Number of users K", ylabel="Time (s, log scale)", title="Computation time")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8)
    return fig

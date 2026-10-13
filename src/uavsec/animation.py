"""Animated replay of a learned UAV trajectory (for slides).

Left: area map with users, eavesdroppers, the UAV trail, and one beam line per
user whose width is proportional to the power ||w_k||^2 the GNN allocates.
Right: per-user secrecy rates (top) and the sum secrecy rate over time (bottom).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib import animation  # noqa: E402

from .channel import Scenario, channels  # noqa: E402
from .config import SystemConfig  # noqa: E402
from .models import Beamformer  # noqa: E402
from .plotting import INK, MUTED, PALETTE, USER_COLOR, draw_nodes  # noqa: E402
from .secrecy import secrecy_rates  # noqa: E402
from .utils import make_generator  # noqa: E402


@torch.no_grad()
def replay(positions: np.ndarray, scenario: Scenario, model: Beamformer, sys: SystemConfig,
           seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-step beam powers ||w_k||^2 and per-user secrecy rates along the trajectory, each (T, K)."""
    device = scenario.users.device
    uav = torch.as_tensor(positions, dtype=torch.float32, device=device)
    h, h_e = channels(sys, uav, scenario.expand(len(positions)), make_generator(seed, device))
    w = model(h, h_e, sys.power_max_w)
    return w.abs().square().sum(-1).cpu().numpy(), secrecy_rates(h, h_e, w).cpu().numpy()


def animate_trajectory(positions: np.ndarray, users: np.ndarray, eves: np.ndarray, beam_power: np.ndarray,
                       secrecy: np.ndarray, area: float, power_max: float, out: Path, fps: int = 10) -> list[Path]:
    k = users.shape[0]
    total = secrecy.sum(axis=1)
    fig = plt.figure(figsize=(10, 5))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1])
    ax_map, ax_bar, ax_sum = fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])

    draw_nodes(ax_map, users, eves)
    ax_map.set(xlim=(0, area), ylim=(0, area), aspect="equal", xlabel="x (m)", ylabel="y (m)")
    ax_map.grid(False)
    beams = [ax_map.plot([], [], color=PALETTE[0], alpha=0.6, solid_capstyle="round")[0] for _ in range(k)]
    trail, = ax_map.plot([], [], ls=":", color=INK, lw=1.4)
    uav, = ax_map.plot([], [], marker="^", ms=12, color=PALETTE[6], ls="none", label="UAV")
    ax_map.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8)

    bars = ax_bar.bar(np.arange(1, k + 1), np.zeros(k), color=USER_COLOR, width=0.6)
    ax_bar.set(xlabel="User k", ylabel="Secrecy rate\n(bps/Hz)", ylim=(0, max(1e-3, secrecy.max()) * 1.1),
               xticks=np.arange(1, k + 1))
    ax_bar.grid(axis="x", visible=False)

    ax_sum.plot(np.arange(len(total)), total, color=MUTED, alpha=0.3, lw=1)
    line, = ax_sum.plot([], [], color=PALETTE[0])
    ax_sum.set(xlabel="Step", ylabel="Sum secrecy rate\n(bps/Hz)", xlim=(0, len(total) - 1))
    title = fig.suptitle("")
    fig.tight_layout()

    def update(t: int):
        x, y = positions[t]
        for kk, beam in enumerate(beams):
            beam.set_data([x, users[kk, 0]], [y, users[kk, 1]])
            beam.set_linewidth(0.3 + 8.0 * beam_power[t, kk] / power_max)
        trail.set_data(positions[: t + 1, 0], positions[: t + 1, 1])
        uav.set_data([x], [y])
        for bar, value in zip(bars, secrecy[t]):
            bar.set_height(value)
        line.set_data(np.arange(t + 1), total[: t + 1])
        title.set_text(f"Step {t}  |  UAV at ({x:.0f}, {y:.0f}) m  |  sum secrecy {total[t]:.2f} bps/Hz")
        return [*beams, trail, uav, *bars, line, title]

    anim = animation.FuncAnimation(fig, update, frames=len(positions), interval=1000 / fps, blit=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = [out.with_suffix(".gif")]
    anim.save(written[0], writer=animation.PillowWriter(fps=fps))
    try:                                   # MP4 when an ffmpeg binary is available (imageio-ffmpeg ships one)
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        anim.save(out.with_suffix(".mp4"), writer=animation.FFMpegWriter(fps=fps, bitrate=2400))
        written.append(out.with_suffix(".mp4"))
    except (ImportError, RuntimeError, FileNotFoundError):
        pass
    plt.close(fig)
    return written

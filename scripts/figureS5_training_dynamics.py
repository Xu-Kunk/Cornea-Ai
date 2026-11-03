"""Figure S5 - training dynamics of the CE-naive and ordinal heads.

``core_claim``
    Three seeds of each arm converge smoothly over three epochs: every training
    loss falls monotonically across the three epoch boundaries, and no seed
    diverges, so neither the 31-class cross-entropy objective nor the ordinal
    cumulative-threshold loss destabilises optimisation over this budget.

``hero_panel``
    The two panels are read together; panel B (ordinal head) is the hero
    because the ordinal cumulative-threshold objective is the one whose
    optimisation behaviour is being established.

``stats_shown``
    none. Every curve is the raw per-step training loss emitted by the Hugging
    Face trainer; no smoothing, interval or test is applied.

``comparability``
    The two arms optimise *different objectives* - cross-entropy over 31
    ``author_cell_type`` classes versus the ordinal cumulative-threshold loss
    on epithelial cells - so their loss magnitudes are not comparable. Only the
    trajectory shape, the epoch count and the convergence point may be compared.
    Each panel therefore carries its own y axis; a shared axis would invite the
    magnitude comparison that the objectives do not license.

``source_data``
    ``save/caft_gf_lam0.0_seed{42,123,456}_*/model/checkpoint-*/trainer_state.json``
    (CE-naive) and
    ``save/ordinal_gf_lam0.0_seed{42,123,456}_*/model/checkpoint-*/trainer_state.json``
    (ordinal head) - the final checkpoint of each of the six runs. Both globs
    are bounded to ``lambda = 0`` of their own run directory.

One figure, one producer: this script is the only writer of
``figures/paper1/figureS5_training_dynamics.{pdf,png}``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import paper1_figure_data as D  # noqa: E402
import paper1_figure_style as S  # noqa: E402

OUT_NAME = "figureS5_training_dynamics"
CANVAS_MM = (S.DOUBLE_COLUMN_MM, 78.0)

SEEDS = (42, 123, 456)

# Canonical arm key -> run-directory stem of the locked training run. Both arms
# are the lambda = 0 runs: CE-naive is 31-class cross-entropy, the ordinal head
# is the cumulative-threshold loss on the four epithelial stages.
ARM_RUNS = {
    "ce_naive": "caft_gf_lam0.0_seed",
    "ordinal": "ordinal_gf_lam0.0_seed",
}

# Shared training protocol of the two arms (asserted, never assumed).
TRAIN_ROWS = 79684
STEPS_PER_EPOCH = 2491
EPOCHS = 3
TOTAL_STEPS = STEPS_PER_EPOCH * EPOCHS  # 7,473

RUN_DIR_RE = re.compile(
    r"^(?P<stem>caft_gf_lam0\.0_seed|ordinal_gf_lam0\.0_seed)(?P<seed>\d+)_"
)
CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")

# Seeds differ by line style and marker, never by hue: the hue is the arm.
SEED_STYLES = (("dotted", "o"), ("dashed", "s"), ("dashdot", "^"))

LINE_WIDTH = 1.0
MARKER_SIZE = 2.0
BOUNDARY_COLOR = "#8A8A8A"


def _final_checkpoints() -> dict[tuple[str, int], Path]:
    """Map ``(arm_key, seed) -> trainer_state.json`` of the final checkpoint.

    Raises
    ------
    FileNotFoundError
        If the bounded glob matches no run for a required (arm, seed).
    """
    runs: dict[tuple[str, int], Path] = {}
    for arm_key, stem in ARM_RUNS.items():
        pattern = f"save/{stem}*/model/checkpoint-*/trainer_state.json"
        for state_path in sorted(S.PROJECT_ROOT.glob(pattern)):
            run_dir = state_path.parents[2].name
            match = RUN_DIR_RE.match(run_dir)
            if match is None or match.group("stem") != stem:
                continue
            seed = int(match.group("seed"))
            if seed not in SEEDS:
                continue
            checkpoint = state_path.parent.name
            checkpoint_match = CHECKPOINT_RE.match(checkpoint)
            if checkpoint_match is None:
                continue
            step = int(checkpoint_match.group(1))
            key = (arm_key, seed)
            previous = runs.get(key)
            if previous is None or step > int(
                CHECKPOINT_RE.match(previous.parent.name).group(1)
            ):
                runs[key] = state_path

    missing = [
        (arm, seed) for arm in ARM_RUNS for seed in SEEDS if (arm, seed) not in runs
    ]
    if missing:
        raise FileNotFoundError(
            f"missing trainer_state.json for runs {missing} under "
            f"{S.PROJECT_ROOT / 'save'}"
        )
    return runs


def _run_history(state_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(steps, losses, steps_per_epoch)`` for one run.

    The step budget comes from ``max_steps`` (the trainer's own count) so the
    epoch boundaries are exact; the epoch count is never defaulted.

    Raises
    ------
    KeyError
        If the run does not record its epoch count or its step budget; a
        missing count must never silently default to one epoch.
    AssertionError
        If the run does not follow the shared three-epoch, 7,473-step
        protocol that the caption states for both arms.
    """
    state = D.load_json(state_path)
    if "num_train_epochs" not in state or state["num_train_epochs"] in (None, 0):
        raise KeyError(
            f"{state_path}: num_train_epochs missing; refusing to default"
        )
    epochs = float(state["num_train_epochs"])
    if "max_steps" not in state or not state["max_steps"]:
        raise KeyError(f"{state_path}: max_steps missing; refusing to default")
    total_steps = float(state["max_steps"])
    history = [entry for entry in state.get("log_history", []) if "loss" in entry]
    if not history:
        raise KeyError(f"{state_path}: no loss records in log_history")
    steps = np.asarray([entry["step"] for entry in history], dtype=float)
    losses = np.asarray([entry["loss"] for entry in history], dtype=float)
    if epochs != EPOCHS or total_steps != TOTAL_STEPS:
        raise AssertionError(
            f"{state_path}: {epochs:g} epochs / {total_steps:g} steps does not "
            f"match the stated protocol ({EPOCHS} epochs / {TOTAL_STEPS} steps)"
        )
    steps_per_epoch = total_steps / epochs
    return steps, losses, steps_per_epoch


def _seed_proxy(colour: str, linestyle: str, marker: str):
    """Legend proxy line for one seed (same arm hue, distinct style)."""
    from matplotlib.lines import Line2D

    return Line2D(
        [0],
        [0],
        color=colour,
        linestyle=linestyle,
        linewidth=LINE_WIDTH,
        marker=marker,
        markersize=MARKER_SIZE,
        markeredgewidth=0.0,
    )


def main() -> int:
    """Render the figure and append its manifest entry."""
    S.set_style()
    runs = _final_checkpoints()
    histories = {}
    steps_per_epoch = None
    for key, state_path in runs.items():
        histories[key] = _run_history(state_path)
        if steps_per_epoch is None:
            steps_per_epoch = histories[key][2]
        elif not np.isclose(
            steps_per_epoch, histories[key][2], rtol=0.0, atol=1e-6
        ):
            raise AssertionError("steps/epoch disagree across arms")

    fig, axes = plt.subplots(
        1, len(ARM_RUNS), figsize=S.figsize(*CANVAS_MM),
        gridspec_kw={"wspace": 0.24},
    )
    handles = []
    labels = []
    for panel_index, arm_key in enumerate(ARM_RUNS):
        ax = axes[panel_index]
        colour = S.ARM_COLORS[arm_key]
        for style_index, seed in enumerate(SEEDS):
            steps, losses, _run_steps = histories[(arm_key, seed)]
            linestyle, marker = SEED_STYLES[style_index]
            ax.plot(
                steps,
                losses,
                color=colour,
                linestyle=linestyle,
                linewidth=LINE_WIDTH,
                marker=marker,
                markersize=MARKER_SIZE,
                markevery=max(1, steps.size // 60),
                markeredgewidth=0.0,
                zorder=3,
            )
            if panel_index == 0:
                handles.append(_seed_proxy(colour, linestyle, marker))
                labels.append(f"seed {seed}")
        for boundary in (steps_per_epoch, 2.0 * steps_per_epoch):
            ax.axvline(boundary, color=BOUNDARY_COLOR, linestyle="--",
                       linewidth=0.7, zorder=0)
        ax.set_xlim(0.0, TOTAL_STEPS * 1.02)
        ax.set_xlabel("Training step")
        ax.set_title(S.ARMS[arm_key])
        # Independent y axes: the two objectives have different loss scales, so
        # a shared axis would distort the shape of the ordinal trajectory.
        ax.set_ylabel("Train loss")

    S.panel_label(axes[0], "A")
    S.panel_label(axes[1], "B")
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(SEEDS),
        frameon=False,
        handlelength=2.6,
        columnspacing=1.4,
    )
    fig.get_layout_engine().set(rect=(0.02, 0.12, 0.99, 0.86))

    pdf_path, png_path = S.save_figure(
        fig,
        OUT_NAME,
        outdir=str(S.FIGURE_DIR),
        inputs={path.parents[2].name: path for path in runs.values()},
    )
    plt.close(fig)
    print(f"[{OUT_NAME}] wrote {pdf_path} and {png_path}")
    print(
        f"[{OUT_NAME}] {len(runs)} runs; {TRAIN_ROWS:,} rows; "
        f"{steps_per_epoch:.0f} steps/epoch; {TOTAL_STEPS:,} steps"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

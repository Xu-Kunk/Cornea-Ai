"""Unit tests for the this study method-arm eval script + Phase B controller v2.

CPU-only, no model loading, no dataset access. Covers the pieces that make
the autonomous Phase A -> B pipeline work:

(a) run_method_arms_eval checkpoint discovery: the LATEST save dir carrying
    the METHOD_ARM_DONE marker wins (a newer marker-less dir, e.g. a crashed
    retrain, is ignored); checkpoint-7473 is preferred, else the highest
    numeric step;
(b) run_method_arms_eval --runs parsing (default all 5 Phase A runs, custom
    arm:lam:seed triples, unknown arm rejected);
(c) run_method_arms_eval runs_status bookkeeping: eval -> "OK" for the run,
    other runs untouched, missing entry appended;
(d) run_phase_b_controller.wait_phase_a_done v2: returns as soon as the 5
    Phase A result JSONs exist (the new completion contract — the train-only
    launcher never flips runs_status eval to "OK").
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import (  # noqa: E402
    run_method_arms_eval as eval_script,
)
from scripts import (  # noqa: E402
    run_phase_b_controller as controller,
)

# ---------------------------------------------------------------------------
# (a) checkpoint discovery
# ---------------------------------------------------------------------------

def test_find_done_save_dir_prefers_latest_dir_with_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(eval_script, "SAVE_DIR", tmp_path)
    done_dir = tmp_path / "method_axis_ft_lam0.5_seed42_20260808-111111"
    done_dir.mkdir()
    (done_dir / "METHOD_ARM_DONE").write_text("2026-08-08T11:11:11Z")
    # a NEWER dir from a crashed retrain: no marker, must be ignored
    crash_dir = tmp_path / "method_axis_ft_lam0.5_seed42_20260809-222222"
    crash_dir.mkdir()
    (crash_dir / "RUN_STATUS.txt").write_text("START ...")

    found = eval_script.find_done_save_dir("axis_ft", 0.5, 42)
    assert found == done_dir


def test_find_done_save_dir_none_when_no_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eval_script, "SAVE_DIR", tmp_path)
    d = tmp_path / "method_flow_lam0.5_seed42_20260809-000000"
    d.mkdir()
    assert eval_script.find_done_save_dir("flow", 0.5, 42) is None


def test_pick_checkpoint_prefers_final_and_falls_back_to_highest_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(eval_script, "SAVE_DIR", tmp_path)
    save_dir = tmp_path / "method_axis_ft_lam1.0_seed42_20260808-213948"
    model_dir = save_dir / "model"
    (model_dir / "checkpoint-4982").mkdir(parents=True)
    (model_dir / "checkpoint-7473").mkdir()

    # prefer the locked final checkpoint
    ck = eval_script.pick_checkpoint(save_dir)
    assert ck is not None
    ckpt, step = ck
    assert ckpt.name == "checkpoint-7473"
    assert step == 7473

    # fallback: highest NUMERIC step (lexicographic would pick checkpoint-99)
    (model_dir / "checkpoint-99").mkdir()
    save_dir2 = tmp_path / "other"
    m2 = save_dir2 / "model"
    (m2 / "checkpoint-99").mkdir(parents=True)
    (m2 / "checkpoint-1000").mkdir()
    ck2 = eval_script.pick_checkpoint(save_dir2)
    assert ck2 is not None
    ckpt2, step2 = ck2
    assert ckpt2.name == "checkpoint-1000"
    assert step2 == 1000


# ---------------------------------------------------------------------------
# (b) --runs parsing
# ---------------------------------------------------------------------------

def test_parse_runs_default_and_custom() -> None:
    default = eval_script._parse_runs(None)
    assert default == eval_script.DEFAULT_RUNS
    assert len(default) == 5

    custom = eval_script._parse_runs("axis_ft:0.5:42,flow:1.0:123")
    assert custom == [("axis_ft", 0.5, 42), ("flow", 1.0, 123)]

    with pytest.raises(ValueError):
        eval_script._parse_runs("nope:0.5:42")
    with pytest.raises(ValueError):
        eval_script._parse_runs("axis_ft:0.5")  # wrong arity


# ---------------------------------------------------------------------------
# (c) runs_status bookkeeping
# ---------------------------------------------------------------------------

def test_mark_status_eval_ok_flips_only_target_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = tmp_path / "runs_status.json"
    monkeypatch.setattr(eval_script, "STATUS_JSON", status)
    status.write_text(json.dumps({
        "grid": ["axis_ft_lam0.5_seed42", "flow_lam0.5_seed42"],
        "mode": "train",
        "runs": [
            {"arm": "axis_ft", "lambda": 0.5, "seed": 42,
             "train": "OK", "eval": "pending", "save_dir": "/tmp/a"},
            {"arm": "flow", "lambda": 0.5, "seed": 42,
             "train": "FAIL", "eval": "skipped", "save_dir": None},
        ],
    }))

    eval_script.mark_status_eval_ok("axis_ft", 0.5, 42, Path("/tmp/a"))
    payload = json.loads(status.read_text())
    runs = {r["arm"]: r for r in payload["runs"]}
    assert runs["axis_ft"]["eval"] == "OK"
    assert runs["flow"]["eval"] == "skipped"  # untouched

    # missing run entry is appended (idempotent robustness)
    eval_script.mark_status_eval_ok("soft_spearman", 1.0, 42, Path("/tmp/ss"))
    payload = json.loads(status.read_text())
    by_arm = {r["arm"]: r for r in payload["runs"]}
    assert by_arm["soft_spearman"]["eval"] == "OK"
    assert by_arm["soft_spearman"]["save_dir"] == "/tmp/ss"
    # the pre-existing flow FAIL entry is untouched (eval flipped only for
    # the target run / appended runs)
    assert by_arm["flow"]["save_dir"] is None
    assert by_arm["flow"]["eval"] == "skipped"


# ---------------------------------------------------------------------------
# (d) controller v2: wait for the 5 result JSONs
# ---------------------------------------------------------------------------

def test_controller_wait_phase_a_done_returns_when_all_results_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(controller, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(controller, "PHASE_A_TIMEOUT_S", 10)
    monkeypatch.setattr(controller, "POLL_S", 0.05)
    for arm, lam, seed in controller.PHASE_A_RUNS:
        (tmp_path / f"results_{arm}_{lam}_{seed}.json").write_text("{}")

    out = controller.wait_phase_a_done()
    assert len(out["phase_a_runs"]) == 5
    assert "results_flow_0.5_42.json" in {p.name for p in tmp_path.glob("results_*.json")}

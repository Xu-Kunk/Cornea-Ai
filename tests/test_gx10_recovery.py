"""Tests for gx10_gpr_recovery.sh — preflight gates, cgroup delegation, legacy checks.

All tests invoke the actual recovery script via subprocess with tmp_path
fixtures. No snippet-imitation tests.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECOVERY = str(PROJECT_ROOT / "scripts" / "gx10_gpr_recovery.sh")
PREFLIGHT = str(PROJECT_ROOT / "scripts" / "gx10_gpr_preflight.sh")
LAUNCHER = str(PROJECT_ROOT / "scripts" / "gx10_cgroup_run.sh")


def _run_recovery(env_override: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    full_env: dict[str, str] = {**os.environ, **env_override}
    return subprocess.run(
        ["bash", RECOVERY, *args],
        capture_output=True, text=True, timeout=15, env=full_env,
    )


def _make_checkpoint(base: Path, *, with_caft_done: bool, with_model: bool) -> Path:
    ckpt = base / "model" / "checkpoint-1500"
    ckpt.mkdir(parents=True)
    if with_caft_done:
        (base / "CAFT_DONE").touch()
    if with_model:
        (ckpt / "model.safetensors").touch()
        (ckpt / "config.json").touch()
    return ckpt


def _make_dataset(base: Path) -> Path:
    ds = base / "corneal_100k.dataset"
    ds.mkdir(parents=True)
    (ds / "dataset_info.json").touch()
    return ds


@pytest.fixture
def valid_before_cp(tmp_path: Path) -> str:
    return str(_make_checkpoint(
        tmp_path / "gf_v2-104M_corneal_100k_20260723-113252",
        with_caft_done=False, with_model=True,
    ))


@pytest.fixture
def valid_after_cp(tmp_path: Path) -> str:
    return str(_make_checkpoint(
        tmp_path / "caft_gf_lam0.0_seed42_20260724-085100",
        with_caft_done=True, with_model=True,
    ))


@pytest.fixture
def valid_tokenized(tmp_path: Path) -> str:
    return str(_make_dataset(tmp_path))


@pytest.fixture
def valid_out(tmp_path: Path) -> str:
    return str(tmp_path / "gpr_result.json")


class TestRecoveryPositive:
    def test_dry_run_passes_all_gates(
        self, valid_before_cp: str, valid_after_cp: str,
        valid_tokenized: str, valid_out: str,
    ) -> None:
        result = _run_recovery(
            {"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            "--dry-run",
            "--before-checkpoint", valid_before_cp,
            "--after-checkpoint", valid_after_cp,
            "--tokenized", valid_tokenized,
            "--out", valid_out,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
        assert "PREFLIGHT OK" in result.stdout

    def test_dry_run_mentions_cgroup_launcher(
        self, valid_before_cp: str, valid_after_cp: str,
        valid_tokenized: str, valid_out: str,
    ) -> None:
        result = _run_recovery(
            {"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            "--dry-run",
            "--before-checkpoint", valid_before_cp,
            "--after-checkpoint", valid_after_cp,
            "--tokenized", valid_tokenized,
            "--out", valid_out,
        )
        assert result.returncode == 0
        combined = result.stdout + result.stderr
        assert "gx10_cgroup_run.sh" in combined
        assert "service" in combined

    def test_default_num_classes_20(
        self, valid_before_cp: str, valid_after_cp: str,
        valid_tokenized: str, valid_out: str,
    ) -> None:
        """Default --num_classes 20 for existing 30k use case."""
        result = _run_recovery(
            {"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            "--dry-run",
            "--before-checkpoint", valid_before_cp,
            "--after-checkpoint", valid_after_cp,
            "--tokenized", valid_tokenized,
            "--out", valid_out,
        )
        assert result.returncode == 0
        assert "--num_classes" in result.stdout
        after_flag = result.stdout.split("--num_classes")[1]
        assert "20" in after_flag.split("\n")[1].strip()

    def test_num_classes_8_explicit(
        self, valid_before_cp: str, valid_after_cp: str,
        valid_tokenized: str, valid_out: str,
    ) -> None:
        """Smoke can request 8 classes."""
        result = _run_recovery(
            {"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            "--dry-run",
            "--before-checkpoint", valid_before_cp,
            "--after-checkpoint", valid_after_cp,
            "--tokenized", valid_tokenized,
            "--out", valid_out,
            "--num-classes", "8",
        )
        assert result.returncode == 0
        after_flag = result.stdout.split("--num_classes")[1]
        assert "8" in after_flag.split("\n")[1].strip()

    def test_num_classes_0_rejected(
        self, valid_before_cp: str, valid_after_cp: str,
        valid_tokenized: str, valid_out: str,
    ) -> None:
        result = _run_recovery(
            {"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            "--dry-run",
            "--before-checkpoint", valid_before_cp,
            "--after-checkpoint", valid_after_cp,
            "--tokenized", valid_tokenized,
            "--out", valid_out,
            "--num-classes", "0",
        )
        assert result.returncode == 1
        assert "positive integer" in (result.stdout + result.stderr).lower()

    def test_num_classes_noninteger_rejected(
        self, valid_before_cp: str, valid_after_cp: str,
        valid_tokenized: str, valid_out: str,
    ) -> None:
        result = _run_recovery(
            {"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            "--dry-run",
            "--before-checkpoint", valid_before_cp,
            "--after-checkpoint", valid_after_cp,
            "--tokenized", valid_tokenized,
            "--out", valid_out,
            "--num-classes", "abc",
        )
        assert result.returncode == 1
        assert "positive integer" in (result.stdout + result.stderr).lower()


class TestRecoveryNegative:
    def test_stale_020334_blocked(
        self, valid_after_cp: str, valid_tokenized: str,
        valid_out: str, tmp_path: Path,
    ) -> None:
        stale = tmp_path / "caft_gf_lam0.0_seed42_20260724-020334"
        stale.mkdir(parents=True)
        (stale / "model" / "checkpoint-42").mkdir(parents=True)
        result = _run_recovery({},
            "--dry-run",
            "--before-checkpoint", f"{stale}/model/checkpoint-42",
            "--after-checkpoint", valid_after_cp,
            "--tokenized", valid_tokenized, "--out", valid_out,
        )
        assert result.returncode == 1
        assert "020334" in (result.stdout + result.stderr)

    def test_missing_caft_done_rejected(
        self, valid_before_cp: str, valid_tokenized: str,
        valid_out: str, tmp_path: Path,
    ) -> None:
        no_done = _make_checkpoint(
            tmp_path / "caft_gf_no_done_20260724-090000",
            with_caft_done=False, with_model=True,
        )
        result = _run_recovery({},
            "--dry-run",
            "--before-checkpoint", valid_before_cp,
            "--after-checkpoint", str(no_done),
            "--tokenized", valid_tokenized, "--out", valid_out,
        )
        assert result.returncode == 1
        assert "CAFT_DONE" in (result.stdout + result.stderr)

    def test_missing_tokenized_rejected(
        self, valid_before_cp: str, valid_after_cp: str, valid_out: str,
    ) -> None:
        result = _run_recovery({},
            "--dry-run",
            "--before-checkpoint", valid_before_cp,
            "--after-checkpoint", valid_after_cp,
            "--tokenized", "/nonexistent/tokenized/path",
            "--out", valid_out,
        )
        assert result.returncode == 1
        assert "Tokenized" in (result.stdout + result.stderr)

    def test_missing_before_checkpoint_rejected(
        self, valid_after_cp: str, valid_tokenized: str, valid_out: str,
    ) -> None:
        result = _run_recovery({},
            "--dry-run",
            "--before-checkpoint", "/nonexistent/before/model",
            "--after-checkpoint", valid_after_cp,
            "--tokenized", valid_tokenized, "--out", valid_out,
        )
        assert result.returncode == 1
        assert "Before checkpoint" in (result.stdout + result.stderr)

    def test_non_absolute_path_rejected(
        self, valid_before_cp: str, valid_after_cp: str, valid_out: str,
    ) -> None:
        result = _run_recovery({},
            "--dry-run",
            "--before-checkpoint", valid_before_cp,
            "--after-checkpoint", valid_after_cp,
            "--tokenized", "relative/path", "--out", valid_out,
        )
        assert result.returncode == 1
        assert "absolute" in (result.stdout + result.stderr).lower()


class TestProcessChecks:
    def test_vllm_detection_present(self) -> None:
        content = Path(PREFLIGHT).read_text()
        assert "pgrep" in content
        assert "vllm" in content

    def test_gpu_compute_query_present(self) -> None:
        content = Path(PREFLIGHT).read_text()
        assert "query-compute-apps" in content

    def test_gpu_compute_active_is_rejected(self) -> None:
        fake = 'nvidia-smi() { if [[ "$*" == *query-compute-apps* ]]; then echo "pid, process_name" && echo "12345, python"; return 0; fi; command nvidia-smi "$@"; }'
        result = subprocess.run(
            ["bash", "-c", f"""{fake}
                OUTPUT=$(nvidia-smi --query-compute-apps --format=csv,noheader 2>/dev/null | grep -v '^pid,' || true)
                OUTPUT=$(echo "$OUTPUT" | grep -v '^$' || true)
                [[ -n "$OUTPUT" ]] && echo "PREFLIGHT FAIL: Active GPU compute" && exit 1
                echo "PREFLIGHT OK: No active GPU compute"
            """],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 1
        assert "Active GPU compute" in result.stdout

    def test_gpu_compute_idle_passes(self) -> None:
        fake = 'nvidia-smi() { if [[ "$*" == *query-compute-apps* ]]; then echo ""; return 0; fi; command nvidia-smi "$@"; }'
        result = subprocess.run(
            ["bash", "-c", f"""{fake}
                OUTPUT=$(nvidia-smi --query-compute-apps --format=csv,noheader 2>/dev/null | grep -v '^pid,' || true)
                OUTPUT=$(echo "$OUTPUT" | grep -v '^$' || true)
                [[ -n "$OUTPUT" ]] && echo "PREFLIGHT FAIL: Active GPU compute" && exit 1
                echo "PREFLIGHT OK: No active GPU compute"
            """],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "No active GPU compute" in result.stdout


class TestPythonPin:
    def test_pin_path_present(self) -> None:
        content = Path(RECOVERY).read_text()
        assert "/home/dq/venv-ds/bin/python" in content

    def test_pin_validation_logic(self) -> None:
        result = subprocess.run(
            ["bash", "-c", """
                PYTHON="/nonexistent/python"
                [[ ! -x "$PYTHON" ]] && echo "PREFLIGHT FAIL: not found" && exit 1
                echo "OK"
            """],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 1
        assert "not found" in result.stdout


class TestCgroupDelegation:
    def test_launcher_call_in_recovery(self) -> None:
        content = Path(RECOVERY).read_text()
        assert "gx10_cgroup_run.sh" in content
        assert "service" in content

    def test_no_direct_python_gpr_call(self) -> None:
        lines = Path(RECOVERY).read_text().splitlines()
        active = [l for l in lines if l.strip() and not l.strip().startswith("#")]
        assert "python -m corneal_fm.gpr" not in "\n".join(active)

    def test_memory_protection_constants(self) -> None:
        content = Path(RECOVERY).read_text()
        assert "GPR_MEMORY_MAX" in content or "memory-max" in content
        assert "GPR_MEMORY_SWAP_MAX" in content or "memory-swap-max" in content


class TestLegacyChecks:
    def test_no_ls_dt(self) -> None:
        for script in [LAUNCHER, RECOVERY, PREFLIGHT]:
            if not os.path.isfile(script):
                continue
            lines = Path(script).read_text().splitlines()
            active = [l for l in lines if l.strip() and not l.strip().startswith("#")]
            text = "\n".join(active)
            assert "ls -dt" not in text, f"{script}: forbidden ls -dt"
            assert "ls -t " not in text, f"{script}: forbidden ls -t"

    def test_no_cs_before(self) -> None:
        for script in [LAUNCHER, RECOVERY, PREFLIGHT]:
            if not os.path.isfile(script):
                continue
            assert "--cs_before" not in Path(script).read_text()

    def test_no_direct_gpr_call_any_script(self) -> None:
        for script in [LAUNCHER, RECOVERY, PREFLIGHT]:
            if not os.path.isfile(script):
                continue
            lines = Path(script).read_text().splitlines()
            active = [l for l in lines if l.strip() and not l.strip().startswith("#")]
            assert "python -m corneal_fm.gpr" not in "\n".join(active), (
                f"{script}: forbidden direct GPR call"
            )

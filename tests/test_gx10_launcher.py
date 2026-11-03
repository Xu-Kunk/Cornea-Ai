"""Tests for gx10_cgroup_run.sh — mode selection, command rendering, scope rejection.

All tests invoke the actual launcher script via subprocess in --dry-run mode.
No GPU, sudo, or systemd execution.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = str(PROJECT_ROOT / "scripts" / "gx10_cgroup_run.sh")


def _run_launcher(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", LAUNCHER, *args],
        capture_output=True, text=True, timeout=15,
    )


class TestServiceMode:
    def test_renders_systemd_run(self) -> None:
        result = _run_launcher("--mode", "service", "--dry-run", "--", "echo", "hello")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "systemd-run" in result.stdout

    def test_includes_oom_score_adjust(self) -> None:
        result = _run_launcher(
            "--mode", "service", "--oom-score-adjust", "500",
            "--dry-run", "--", "echo", "hello",
        )
        assert "OOMScoreAdjust=500" in result.stdout

    def test_includes_memory_max(self) -> None:
        result = _run_launcher(
            "--mode", "service", "--memory-max", "100G",
            "--dry-run", "--", "echo", "hello",
        )
        assert "MemoryMax=100G" in result.stdout

    def test_includes_memory_swap_max(self) -> None:
        result = _run_launcher(
            "--mode", "service", "--memory-swap-max", "0",
            "--dry-run", "--", "echo", "hello",
        )
        assert "MemorySwapMax=0" in result.stdout

    def test_includes_runtime_max_sec(self) -> None:
        result = _run_launcher(
            "--mode", "service", "--runtime-max", "86400",
            "--dry-run", "--", "echo", "hello",
        )
        assert "RuntimeMaxSec" in result.stdout

    def test_includes_timeout_stop_sec(self) -> None:
        result = _run_launcher(
            "--mode", "service", "--runtime-max", "86400",
            "--dry-run", "--", "echo", "hello",
        )
        assert "TimeoutStopSec" in result.stdout

    def test_sets_oom_policy(self) -> None:
        result = _run_launcher("--mode", "service", "--dry-run", "--", "echo", "hello")
        assert "OOMPolicy=kill" in result.stdout

    def test_generates_unit_name(self) -> None:
        result = _run_launcher(
            "--mode", "service", "--unit", "gx10-caft-test",
            "--dry-run", "--", "echo", "hello",
        )
        assert "gx10-caft-test" in result.stdout

    def test_allocator_expandable_segments_false(self) -> None:
        result = _run_launcher("--mode", "service", "--dry-run", "--", "echo", "hello")
        assert "expandable_segments:False" in result.stdout
        assert "expandable_segments:True" not in result.stdout

    def test_allocator_value_not_true(self) -> None:
        result = _run_launcher("--mode", "service", "--dry-run", "--", "echo", "hello")
        assert "PYTORCH_CUDA_ALLOC_CONF" in result.stdout
        assert "True" not in result.stdout.split("PYTORCH_CUDA_ALLOC_CONF=")[-1].split()[0]

    def test_with_log_dir(self) -> None:
        result = _run_launcher(
            "--mode", "service", "--log-dir", "/tmp/gx10-logs",
            "--dry-run", "--", "echo", "hello",
        )
        assert "StandardOutput=file:" in result.stdout


class TestScopeMode:
    def test_renders_systemd_run(self) -> None:
        result = _run_launcher("--mode", "scope", "--dry-run", "--", "echo", "hello")
        assert result.returncode == 0
        assert "systemd-run" in result.stdout
        assert "--scope" in result.stdout

    def test_includes_memory_max(self) -> None:
        result = _run_launcher(
            "--mode", "scope", "--memory-max", "100G",
            "--dry-run", "--", "echo", "hello",
        )
        assert "MemoryMax=100G" in result.stdout

    def test_rejects_oom_score_adjust(self) -> None:
        result = _run_launcher(
            "--mode", "scope", "--oom-score-adjust", "500",
            "--dry-run", "--", "echo", "hello",
        )
        assert result.returncode == 1
        combined = (result.stdout + result.stderr).lower()
        assert any(
            kw in combined
            for kw in ["oomscoreadjust", "not supported", "cannot", "property", "exec"]
        ), f"No clear rejection message in: {combined}"


class TestArgumentValidation:
    def test_missing_mode(self) -> None:
        result = _run_launcher("--dry-run", "--", "echo", "hello")
        assert result.returncode != 0

    def test_missing_separator(self) -> None:
        result = _run_launcher("--mode", "service", "--dry-run", "echo", "hello")
        assert result.returncode != 0

    def test_separator_before_command(self) -> None:
        result = _run_launcher("--mode", "service", "--dry-run", "--", "echo", "hello")
        assert result.returncode == 0


class TestDryRunNoExecution:
    def test_does_not_block(self) -> None:
        result = _run_launcher("--mode", "service", "--dry-run", "--", "sleep", "10")
        assert result.returncode == 0

    def test_output_contains_command(self) -> None:
        result = _run_launcher(
            "--mode", "service", "--dry-run", "--", "my-script", "--seed", "42",
        )
        assert "my-script" in result.stdout
        assert result.returncode == 0

#!/usr/bin/env bash
# NEUTERED 2026-07-28: this script auto-relaunched a SECOND grid (nohup run_lambda_grid.py)
# while the original grid was still running, causing duplicate grids -> duplicate caft ->
# GPU contention -> CAFT/GPR kills (rc=-9) -> manual pkills -> relaunch cascade.
# Original preserved at scripts/auto_recover.sh.disabled. Do NOT re-enable.
exit 0

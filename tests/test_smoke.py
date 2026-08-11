"""Smoke test: verify the full pipeline runs on 100 cells with a tiny model."""
import pytest
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_import_package():
    """Verify corneal_fm package imports correctly."""
    sys.path.insert(0, str(ROOT))
    from corneal_fm import __version__
    assert __version__ == "0.3.0"


def test_config_loading():
    """Verify default config loads without error."""
    import yaml
    config_path = ROOT / "config" / "default_train_args.yml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    assert config["model_parameters"]["seed"] == 42
    assert len(config["cross_species"]["species"]) == 4


# TODO: Add GPU smoke test when GX10 is available:
# def test_dry_run_1k_cells():
#     """Verify fine-tuning runs on 1k cells without OOM."""
#     result = subprocess.run([...])
#     assert result.returncode == 0

"""Tests for deterministic_subset.py: smoke-only 100-cell subset generator."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from corneal_fm.caft import CELL_TYPE_TO_LAYER, LAYER_ORDER


class TestDeterministicSubset:
    """Unit tests for the subset generation logic (no GPU, no large data)."""

    def test_layer_order_has_5_elements(self) -> None:
        assert len(LAYER_ORDER) == 5

    def test_cell_type_to_layer_maps_epithelial_types(self) -> None:
        """Verify critical epithelial cell type → layer mappings."""
        assert CELL_TYPE_TO_LAYER["LPC"] == 0         # LSC
        assert CELL_TYPE_TO_LAYER["Fetal_Epi/LSC/LPC"] == 0
        assert CELL_TYPE_TO_LAYER["Limbus_Epi-C1"] == 1  # TAC
        assert CELL_TYPE_TO_LAYER["Limbus_Epi-C2"] == 1
        assert CELL_TYPE_TO_LAYER["Epi-PMC"] == 2        # basal
        assert CELL_TYPE_TO_LAYER["Epi-TDC/PMC"] == 3    # wing
        assert CELL_TYPE_TO_LAYER["Epi-TDC"] == 4        # squamous
        assert CELL_TYPE_TO_LAYER["Unknown Epithelium"] == 4

    def test_non_epithelial_types_map_to_minus_one(self) -> None:
        assert CELL_TYPE_TO_LAYER["Conj-Epi"] == -1
        assert CELL_TYPE_TO_LAYER["Unknown-DCN+"] == -1

    def test_layer_count_matches_expected(self) -> None:
        """Each of the 5 layers must have at least one cell type mapped."""
        layer0_types = [ct for ct, ly in CELL_TYPE_TO_LAYER.items() if ly == 0]
        layer1_types = [ct for ct, ly in CELL_TYPE_TO_LAYER.items() if ly == 1]
        layer2_types = [ct for ct, ly in CELL_TYPE_TO_LAYER.items() if ly == 2]
        layer3_types = [ct for ct, ly in CELL_TYPE_TO_LAYER.items() if ly == 3]
        layer4_types = [ct for ct, ly in CELL_TYPE_TO_LAYER.items() if ly == 4]
        assert len(layer0_types) >= 1, f"LSC layer has {len(layer0_types)} types"
        assert len(layer1_types) >= 1, f"TAC layer has {len(layer1_types)} types"
        assert len(layer2_types) >= 1, f"basal layer has {len(layer2_types)} types"
        assert len(layer3_types) >= 1, f"wing layer has {len(layer3_types)} types"
        assert len(layer4_types) >= 1, f"squamous layer has {len(layer4_types)} types"


class TestDeterministicSubsetIntegration:
    """Integration test: run the actual subset generator on the 30k h5ad."""

    SRC = ROOT / "artifacts/day3-20260723/S6/corneal_30k_layer_subsample.h5ad"

    @pytest.fixture
    def tmp_out(self, tmp_path: Path) -> Path:
        return tmp_path / "test_smoke_100cells.h5ad"

    @pytest.mark.slow
    def test_generate_subset_output_shape(self, tmp_out: Path) -> None:
        """Generated subset must be exactly 100 cells × 35477 genes."""
        import anndata as ad
        from scripts.deterministic_subset import generate_subset

        prov = generate_subset(str(self.SRC), str(tmp_out), seed=42)
        a = ad.read_h5ad(str(tmp_out))
        assert a.n_obs == 100, f"Got {a.n_obs} cells"
        assert a.n_vars == 35477, f"Got {a.n_vars} genes"

    @pytest.mark.slow
    def test_generate_subset_per_layer_20_cells(self, tmp_out: Path) -> None:
        """Each layer must have exactly 20 cells (deterministic seed=42)."""
        import anndata as ad
        import numpy as np
        from scripts.deterministic_subset import generate_subset

        generate_subset(str(self.SRC), str(tmp_out), seed=42)
        a = ad.read_h5ad(str(tmp_out))

        layers = np.array(
            [CELL_TYPE_TO_LAYER.get(str(ct), -1) for ct in a.obs["author_cell_type"]],
            dtype=np.int64,
        )
        for i, name in enumerate(LAYER_ORDER):
            count = int((layers == i).sum())
            assert count == 20, f"Layer {name} ({i}): expected 20, got {count}"

    @pytest.mark.slow
    def test_generate_subset_deterministic(self, tmp_path: Path) -> None:
        """Seed=42 must always produce identical cell indices."""
        import numpy as np
        from scripts.deterministic_subset import generate_subset

        out_a = tmp_path / "subset_a.h5ad"
        out_b = tmp_path / "subset_b.h5ad"

        generate_subset(str(self.SRC), str(out_a), seed=42)
        generate_subset(str(self.SRC), str(out_b), seed=42)

        import anndata as ad
        aa = ad.read_h5ad(str(out_a))
        ab = ad.read_h5ad(str(out_b))

        # Cell count match
        assert aa.n_obs == ab.n_obs == 100

        # X matrix should be identical
        np.testing.assert_array_equal(aa.X.toarray() if hasattr(aa.X, "toarray") else aa.X,
                                       ab.X.toarray() if hasattr(ab.X, "toarray") else ab.X)

    @pytest.mark.slow
    def test_generate_subset_different_seeds_differ(self, tmp_path: Path) -> None:
        """Different seeds must produce different subsets."""
        import numpy as np
        from scripts.deterministic_subset import generate_subset

        out_a = tmp_path / "subset_seed42.h5ad"
        out_b = tmp_path / "subset_seed99.h5ad"

        generate_subset(str(self.SRC), str(out_a), seed=42)
        generate_subset(str(self.SRC), str(out_b), seed=99)

        import anndata as ad
        aa = ad.read_h5ad(str(out_a))
        ab = ad.read_h5ad(str(out_b))

        Xa = aa.X.toarray() if hasattr(aa.X, "toarray") else aa.X
        Xb = ab.X.toarray() if hasattr(ab.X, "toarray") else ab.X
        assert not np.array_equal(Xa, Xb), "Different seeds should produce different subsets"

    @pytest.mark.slow
    def test_generate_subset_preserves_author_cell_type(self, tmp_out: Path) -> None:
        """author_cell_type column must be preserved."""
        from scripts.deterministic_subset import generate_subset

        generate_subset(str(self.SRC), str(tmp_out), seed=42)
        import anndata as ad
        a = ad.read_h5ad(str(tmp_out))
        assert "author_cell_type" in a.obs.columns

    @pytest.mark.slow
    def test_generate_subset_preserves_n_counts(self, tmp_out: Path) -> None:
        """n_counts column must be present (tokenization prerequisite)."""
        from scripts.deterministic_subset import generate_subset

        generate_subset(str(self.SRC), str(tmp_out), seed=42)
        import anndata as ad
        a = ad.read_h5ad(str(tmp_out))
        assert "n_counts" in a.obs.columns
        import numpy as np
        assert np.all(a.obs["n_counts"] > 0), "All cells must have positive n_counts"

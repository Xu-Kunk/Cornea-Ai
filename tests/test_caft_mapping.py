"""Ratified 31->5 layer mapping assertions (Day-3 PI ratification, 2026-07-23).

TDD RED-then-GREEN anchor for the AUTHOR-RATIFIED mapping (see
artifacts/day3-20260723/S3/mapping_ratification.md and
docs/research/virtual-cell/CAFT-Day3-layer-mapping.md).

These tests assert the PI-ratified layer assignments and FAIL against the
pre-ratification code, in which:
  - Epi-PMC  was 4 (squamous)  -> ratified 2 (basal)   [KRT3/12 lower, post-mitotic]
  - Epi-TDC  was 2 (basal)     -> ratified 4 (squamous) [KRT3/12 highest, cornified]
  - Unknown Epithelium was absent (default -1) -> ratified 4 (squamous, PI decision)
The continuity loss depends on this ordering being biologically correct, so the
mapping itself is treated as a tested contract.
"""

from corneal_fm.caft import (
    CELL_TYPE_TO_LAYER,
    LAYER_ORDER,
    NON_LAYER,
    author_cell_type_to_layer,
    class_to_layer_map,
)


def test_layer_order_is_five_and_authoritative() -> None:
    assert LAYER_ORDER == ["LSC", "TAC", "basal", "wing", "squamous"]


def test_stem_and_TAC_layers() -> None:
    assert CELL_TYPE_TO_LAYER["LPC"] == 0
    assert CELL_TYPE_TO_LAYER["Fetal_Epi/LSC/LPC"] == 0
    assert CELL_TYPE_TO_LAYER["Limbus_Epi-C1"] == 1
    assert CELL_TYPE_TO_LAYER["Limbus_Epi-C2"] == 1


def test_corrected_epithelial_order() -> None:
    # PI ratification reverses the prior code: TDC = squamous (highest KRT3/KRT12,
    # SPRR2A/B + IVL cornification), PMC = basal (lower KRT3/KRT12, KRT5+ post-mitotic).
    assert CELL_TYPE_TO_LAYER["Epi-PMC"] == 2
    assert CELL_TYPE_TO_LAYER["Epi-TDC/PMC"] == 3
    assert CELL_TYPE_TO_LAYER["Epi-TDC"] == 4


def test_unknown_epithelium_pi_decision_is_squamous() -> None:
    # PI chose to assign atlas-unsubtyped epithelium (3561 cells) to squamous(4);
    # plausible but unvalidated -- documented in the ratification record.
    assert CELL_TYPE_TO_LAYER.get("Unknown Epithelium") == 4


def test_off_axis_classes_excluded() -> None:
    # Conjunctival epithelium is a distinct lineage (KRT13+/KRT12-) -> off axis.
    assert author_cell_type_to_layer("Conj-Epi") == NON_LAYER == -1
    # Unknown-DCN+ too few / ambiguous (DCN is both LPC and stromal) -> off axis.
    assert author_cell_type_to_layer("Unknown-DCN+") == NON_LAYER
    # Representative non-epithelial classes fall through to the NON_LAYER default.
    assert author_cell_type_to_layer("Corneal Stromal Keratocytes") == NON_LAYER
    assert author_cell_type_to_layer("Macrophages") == NON_LAYER
    assert author_cell_type_to_layer("Vascular Endothelium") == NON_LAYER


def test_class_to_layer_map_is_total_and_deterministic() -> None:
    id_class = {
        0: "LPC",
        1: "Epi-TDC",
        2: "Epi-PMC",
        3: "Conj-Epi",
        4: "Pericytes",
        5: "Unknown Epithelium",
    }
    m = class_to_layer_map(id_class)
    assert m[0] == 0   # LSC
    assert m[1] == 4   # squamous (corrected)
    assert m[2] == 2   # basal (corrected)
    assert m[3] == -1  # conjunctival off-axis
    assert m[4] == -1  # pericyte off-axis
    assert m[5] == 4   # unknown epithelium -> squamous (PI decision)
    assert all(-1 <= v < len(LAYER_ORDER) for v in m.values())

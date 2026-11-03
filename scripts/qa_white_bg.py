#!/usr/bin/env python3
"""Sample interior pixels of generated figures to verify white background."""

from pathlib import Path
from PIL import Image

FIG_DIR = Path("/home/dq/Codex/Ai/figures/paper1")

# Sample points as fractions of (width, height) chosen to land inside plot areas.
SAMPLE_FIGURES = {
    "figure2_layerwise_rho.png": [(0.20, 0.20), (0.50, 0.20), (0.80, 0.20)],
    "figure5_external_eval.png": [(0.20, 0.20), (0.50, 0.20), (0.80, 0.20)],
    "figure7_corn_lora_umap.png": [(0.20, 0.20), (0.55, 0.20), (0.85, 0.20)],
    "figure8_bio_validation.png": [(0.20, 0.15), (0.50, 0.15), (0.20, 0.55), (0.50, 0.55)],
    "figure9_cross_architecture_spectrum.png": [(0.20, 0.20), (0.50, 0.20), (0.80, 0.20)],
    "fig_umap_collapse.png": [(0.20, 0.15), (0.55, 0.15), (0.85, 0.15)],
    "fig_umap_scgpt_bistable.png": [(0.15, 0.20), (0.40, 0.20), (0.65, 0.20)],
    "figureS3_per_seed_reproducibility.png": [(0.20, 0.15), (0.50, 0.15), (0.25, 0.55)],
    "figureS7_layerwise_rho_per_seed.png": [(0.20, 0.20), (0.50, 0.20), (0.80, 0.20)],
}


def main() -> int:
    for fname, points in SAMPLE_FIGURES.items():
        path = FIG_DIR / fname
        if not path.exists():
            print(f"[SKIP] {fname}: not found")
            continue
        im = Image.open(path)
        if im.mode != "RGB":
            im = im.convert("RGB")
        w, h = im.size
        print(f"{fname} ({w}x{h})")
        for fx, fy in points:
            x = int(fx * w)
            y = int(fy * h)
            r, g, b = im.getpixel((x, y))
            ok = (r == 255 and g == 255 and b == 255)
            status = "OK" if ok else "NOT WHITE"
            print(f"  ({x:4d},{y:4d}) RGB=({r:3d},{g:3d},{b:3d}) {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

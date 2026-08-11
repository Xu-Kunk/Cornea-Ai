"""Corneal-FM: Tissue-specific fine-tuning of single-cell foundation models for corneal research.

Corneal-FM adapts large-scale single-cell foundation models (Geneformer, scGPT) to
corneal cell biology through targeted fine-tuning. The package provides:

- **Config management**: YAML-based experiment configuration with CLI overrides
- **Trainer classes**: Geneformer and scGPT fine-tuning with AMP, gradient clipping,
  5-fold cross-validation, and dry-run mode
- **Benchmark suite**: Four-task evaluation (E1 cell classification, E2 perturbation
  prediction, E3 cross-species, E4 embedding analysis)
- **Utility functions**: GPU-aware device detection, logging, timing, parameter counting
"""

from __future__ import annotations

__version__ = "0.3.0"

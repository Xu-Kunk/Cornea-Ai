"""Benchmark suite for Corneal-FM: E1–E4 evaluation of fine-tuned models.

The four evaluation tasks follow the protocols established in
*Geneformer* (Theodoris et al., Nature 2023) and *scGPT* (Cui et al.,
Nature Methods 2024):

- **E1 — Cell classification**: cell-type prediction
- **E2 — Perturbation prediction**: expression change after gene perturbation
- **E3 — Cross-species**: embedding alignment between human and mouse
- **E4 — Embedding analysis**: condition-level separation in latent space
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from corneal_fm.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


class Benchmark:
    """Four-task evaluation suite for single-cell foundation models.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary.
    device : torch.device, optional
        Compute device.  Auto-detected if not provided.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = config or {}
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    # ------------------------------------------------------------------
    # E1 — Cell Classification
    # ------------------------------------------------------------------

    def e1_cell_classification(
        self,
        model: torch.nn.Module,
        test_data: Dataset,
    ) -> dict[str, float]:
        """Evaluate cell-type classification accuracy.

        Parameters
        ----------
        model : torch.nn.Module
            Fine-tuned model in evaluation mode.
        test_data : Dataset
            Test dataset yielding ``(input, label)`` pairs.

        Returns
        -------
        dict[str, float]
            Metrics: ``accuracy``, ``precision``, ``recall``, ``macro_f1``,
            ``kappa``, ``n_samples``.
        """
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            cohen_kappa_score,
        )

        model.eval()
        loader = DataLoader(
            test_data,
            batch_size=self.config.get("batch_size", 64),
            shuffle=False,
            num_workers=self.config.get("num_workers", 4),
        )

        all_preds: list[int] = []
        all_targets: list[int] = []

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch
                else:
                    inputs, targets = batch, batch

                inputs = inputs.to(self.device)
                targets_np = targets.cpu().numpy() if torch.is_tensor(targets) else targets  # type: ignore[no-any-unimported]
                logits = model(inputs)
                preds = logits.argmax(dim=-1).cpu().numpy()

                all_preds.extend(preds.tolist())
                all_targets.extend(
                    targets_np.tolist() if hasattr(targets_np, "tolist") else list(targets_np)
                )

        report = classification_report(
            all_targets, all_preds, output_dict=True, zero_division=0
        )
        macro_avg = report.get("macro avg", {})
        accuracy = accuracy_score(all_targets, all_preds)
        kappa = cohen_kappa_score(all_targets, all_preds)

        results: dict[str, float] = {
            "accuracy": float(accuracy),
            "precision": float(macro_avg.get("precision", 0.0)),
            "recall": float(macro_avg.get("recall", 0.0)),
            "macro_f1": float(macro_avg.get("f1-score", 0.0)),
            "kappa": float(kappa),
            "n_samples": len(all_targets),
        }
        logger.info("E1 Cell Classification: accuracy=%.4f, macro_f1=%.4f", results["accuracy"], results["macro_f1"])
        return results

    # ------------------------------------------------------------------
    # E2 — Perturbation Prediction
    # ------------------------------------------------------------------

    def e2_perturbation_prediction(
        self,
        model: torch.nn.Module,
        test_data: Dataset,
        target_genes: list[str] | None = None,
    ) -> dict[str, float]:
        """Evaluate perturbation prediction (expression change correlation).

        Parameters
        ----------
        model : torch.nn.Module
            Fine-tuned model.
        test_data : Dataset
            Dataset yielding ``(control_expression, perturbed_expression)``
            or ``(input, target)`` pairs.
        target_genes : list[str], optional
            Names of the perturbed genes (used for logging only).

        Returns
        -------
        dict[str, float]
            Metrics: ``pearson_r``, ``p_value``, ``spearman_r``, ``n_genes``.
        """
        from scipy.stats import pearsonr, spearmanr

        model.eval()
        loader = DataLoader(
            test_data,
            batch_size=self.config.get("batch_size", 64),
            shuffle=False,
            num_workers=self.config.get("num_workers", 4),
        )

        all_preds: list[float] = []
        all_targets: list[float] = []

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch
                else:
                    inputs, targets = batch, batch

                inputs = inputs.to(self.device)
                outputs = model(inputs)

                all_preds.extend(outputs.cpu().flatten().tolist())
                targets_np = targets.cpu().flatten() if torch.is_tensor(targets) else targets  # type: ignore[no-any-unimported]
                all_targets.extend(
                    targets_np.tolist() if hasattr(targets_np, "tolist") else targets_np
                )

        # Guard against constant arrays
        if len(set(all_preds)) < 2 or len(set(all_targets)) < 2:
            return {
                "pearson_r": 0.0,
                "p_value": 1.0,
                "spearman_r": 0.0,
                "n_genes": len(all_targets),
            }

        pearson_r, p_val = pearsonr(all_preds, all_targets)
        spearman_r, _ = spearmanr(all_preds, all_targets)

        results: dict[str, float] = {
            "pearson_r": float(pearson_r),
            "p_value": float(p_val),
            "spearman_r": float(spearman_r),
            "n_genes": len(all_targets),
        }
        logger.info("E2 Perturbation: pearson_r=%.4f, p=%.4e", results["pearson_r"], results["p_value"])
        return results

    # ------------------------------------------------------------------
    # E3 — Cross-Species
    # ------------------------------------------------------------------

    def e3_cross_species(
        self,
        model: torch.nn.Module,
        species_data: dict[str, Dataset],
        ortholog_map: dict[str, str] | None = None,
    ) -> dict[str, float]:
        """Evaluate cross-species embedding alignment.

        Parameters
        ----------
        model : torch.nn.Module
            Fine-tuned model.
        species_data : dict[str, Dataset]
            Dictionary mapping species names (e.g. ``"human"``,
            ``"mouse"``) to their datasets.
        ortholog_map : dict[str, str], optional
            Mapping from target-species gene names to reference-species
            gene names.  If ``None``, assumes matched feature order.

        Returns
        -------
        dict[str, float]
            Metrics: ``embedding_cosine_similarity``,
            ``embedding_euclidean_distance``, ``n_genes``.
        """
        model.eval()
        embeddings: dict[str, torch.Tensor] = {}

        with torch.no_grad():
            for species, dataset in species_data.items():
                loader = DataLoader(
                    dataset,
                    batch_size=self.config.get("batch_size", 64),
                    shuffle=False,
                    num_workers=self.config.get("num_workers", 4),
                )
                species_embeds: list[torch.Tensor] = []
                for batch in loader:
                    if isinstance(batch, (tuple, list)):
                        inputs = batch[0]
                    else:
                        inputs = batch
                    inputs = inputs.to(self.device)
                    # Use penultimate layer or model.encode(.) if available
                    emb = self._extract_embeddings(model, inputs)
                    species_embeds.append(emb.cpu())
                embeddings[species] = torch.cat(species_embeds, dim=0)

        if len(embeddings) < 2:
            logger.warning("E3 requires at least two species.")
            return {
                "embedding_cosine_similarity": 0.0,
                "embedding_euclidean_distance": 0.0,
                "n_genes": 0,
            }

        species_names = list(embeddings.keys())
        emb_a = embeddings[species_names[0]]
        emb_b = embeddings[species_names[1]]

        # Align to the smaller set
        min_len = min(emb_a.size(0), emb_b.size(0))
        emb_a, emb_b = emb_a[:min_len], emb_b[:min_len]

        cos_sim = torch.nn.functional.cosine_similarity(emb_a, emb_b, dim=-1).mean().item()
        eucl_dist = torch.norm(emb_a - emb_b, dim=-1).mean().item()

        results: dict[str, float] = {
            "embedding_cosine_similarity": float(cos_sim),
            "embedding_euclidean_distance": float(eucl_dist),
            "n_genes": min_len,
        }
        logger.info("E3 Cross-Species: cos_sim=%.4f, eucl_dist=%.4f", cos_sim, eucl_dist)
        return results

    # ------------------------------------------------------------------
    # E4 — Embedding Analysis
    # ------------------------------------------------------------------

    def e4_embedding_analysis(
        self,
        model: torch.nn.Module,
        data: Dataset,
        condition_labels: list[int] | None = None,
    ) -> dict[str, float]:
        """Evaluate condition-level separation in the embedding space.

        Parameters
        ----------
        model : torch.nn.Module
            Fine-tuned model.
        data : Dataset
            Dataset yielding ``(input, label)`` pairs.  Labels are used
            as condition labels if *condition_labels* is ``None``.
        condition_labels : list[int], optional
            Explicit condition labels.  Overrides dataset labels.

        Returns
        -------
        dict[str, float]
            Metrics: ``silhouette_score``, ``centroid_distance``,
            ``n_conditions``, ``n_cells``.
        """
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import LabelEncoder

        model.eval()
        loader = DataLoader(
            data,
            batch_size=self.config.get("batch_size", 64),
            shuffle=False,
            num_workers=self.config.get("num_workers", 4),
        )

        all_embeds: list[np.ndarray] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch
                else:
                    inputs, targets = batch, batch

                inputs = inputs.to(self.device)
                emb = self._extract_embeddings(model, inputs)
                all_embeds.append(emb.cpu().numpy())

                targets_np = targets.cpu().numpy() if torch.is_tensor(targets) else targets  # type: ignore[no-any-unimported]
                all_labels.extend(
                    targets_np.tolist() if hasattr(targets_np, "tolist") else list(targets_np)
                )

        if not all_embeds:
            return {"silhouette_score": 0.0, "centroid_distance": 0.0, "n_conditions": 0, "n_cells": 0}

        embeddings = np.concatenate(all_embeds, axis=0)
        labels_array = np.array(all_labels)

        # Use provided condition labels or encode dataset labels
        if condition_labels is not None:
            cond = np.array(condition_labels)
        else:
            le = LabelEncoder()
            cond = le.fit_transform(labels_array)

        unique_conds = np.unique(cond)
        n_conditions = len(unique_conds)

        # Silhouette score requires at least 2 clusters and n > n_clusters
        sil = 0.0
        if 2 <= n_conditions < len(cond):
            sil = float(silhouette_score(embeddings, cond, metric="cosine"))

        # Centroid distance: mean pairwise distance between condition centroids
        centroids = []
        for c in unique_conds:
            mask = cond == c
            centroids.append(embeddings[mask].mean(axis=0))
        centroids = np.stack(centroids)

        centroid_dist = 0.0
        if len(centroids) >= 2:
            dists = []
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    d = np.linalg.norm(centroids[i] - centroids[j])
                    dists.append(d)
            centroid_dist = float(np.mean(dists))

        results: dict[str, float] = {
            "silhouette_score": sil,
            "centroid_distance": centroid_dist,
            "n_conditions": n_conditions,
            "n_cells": len(cond),
        }
        logger.info("E4 Embedding: silhouette=%.4f, centroid_dist=%.4f", sil, centroid_dist)
        return results

    # ------------------------------------------------------------------
    # Run All
    # ------------------------------------------------------------------

    def run_all(
        self,
        model: torch.nn.Module,
        datasets: dict[str, Dataset],
        target_genes: list[str] | None = None,
        ortholog_map: dict[str, str] | None = None,
        condition_labels: list[int] | None = None,
    ) -> "pd.DataFrame":
        """Run all four evaluation tasks and return a results table.

        Parameters
        ----------
        model : torch.nn.Module
            Fine-tuned model.
        datasets : dict[str, Dataset]
            Named dataset dictionary.  Expected keys:
            ``"e1_test"``, ``"e2_test"``, ``"e3_human"``, ``"e3_mouse"``,
            ``"e4_data"``.
        target_genes : list[str], optional
            Gene names for perturbation prediction (E2).
        ortholog_map : dict[str, str], optional
            Ortholog mapping for cross-species evaluation (E3).
        condition_labels : list[int], optional
            Explicit condition labels for embedding analysis (E4).

        Returns
        -------
        pd.DataFrame
            Results table with tasks as rows and metrics as columns.
        """
        import pandas as pd  # type: ignore[import-untyped]

        model = model.to(self.device)
        model.eval()

        rows: list[dict[str, Any]] = []

        # E1
        if "e1_test" in datasets and self.config.get("eval_classification", True):
            e1 = self.e1_cell_classification(model, datasets["e1_test"])
            row: dict[str, Any] = {"task": "E1 Cell Classification"}
            row.update(e1)
            rows.append(row)

        # E2
        if "e2_test" in datasets and self.config.get("eval_perturbation", False):
            e2 = self.e2_perturbation_prediction(
                model, datasets["e2_test"], target_genes=target_genes
            )
            row = {"task": "E2 Perturbation Prediction"}
            row.update(e2)
            rows.append(row)

        # E3
        species_data: dict[str, Dataset] = {}
        if "e3_human" in datasets and "e3_mouse" in datasets and self.config.get("eval_cross_species", False):
            species_data["human"] = datasets["e3_human"]
            species_data["mouse"] = datasets["e3_mouse"]
            e3 = self.e3_cross_species(model, species_data, ortholog_map=ortholog_map)
            row = {"task": "E3 Cross-Species"}
            row.update(e3)
            rows.append(row)

        # E4
        if "e4_data" in datasets and self.config.get("eval_embedding", True):
            e4 = self.e4_embedding_analysis(
                model, datasets["e4_data"], condition_labels=condition_labels
            )
            row = {"task": "E4 Embedding Analysis"}
            row.update(e4)
            rows.append(row)

        results_df = pd.DataFrame(rows).set_index("task") if rows else pd.DataFrame()
        logger.info("Benchmark complete: %d tasks evaluated", len(rows))
        return results_df

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_results(
        self,
        results: "pd.DataFrame",
        save_dir: str | Path,
    ) -> None:
        """Generate publication-quality benchmark summary plots.

        Produces:

        - ``benchmark_bars.png``: grouped bar chart of E1 metrics
        - ``benchmark_embedding.png``: UMAP of embeddings with condition
          labels (requires an ``"e4_data"`` entry in the dataset cache or
          a pre-computed embedding).

        Parameters
        ----------
        results : pd.DataFrame
            Results table from :meth:`run_all`.
        save_dir : str | Path
            Output directory for plots.
        """
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # --- Bar plot for E1 ---
        if "E1 Cell Classification" in results.index:
            e1_row = results.loc["E1 Cell Classification"]
            metrics = ["accuracy", "precision", "recall", "macro_f1", "kappa"]
            values = [e1_row.get(m, 0.0) for m in metrics]

            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
            bars = ax.bar(metrics, values, color=colors, edgecolor="white", width=0.6)

            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

            ax.set_ylim(0, 1.1)
            ax.set_ylabel("Score")
            ax.set_title("E1: Cell Classification Performance")
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            fig.savefig(save_dir / "benchmark_bars.png", dpi=300)
            plt.close(fig)
            logger.info("Saved benchmark_bars.png")

        # --- UMAP visualization (if embeddings available) ---
        if hasattr(self, "_cached_embeddings") and self._cached_embeddings is not None:
            self._plot_umap(save_dir)

    def _plot_umap(self, save_dir: Path) -> None:
        """Generate a UMAP embedding scatter plot."""
        try:
            import umap  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("umap-learn not installed. Skipping UMAP plot.")
            return

        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")

        emb = self._cached_embeddings
        reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
        emb_2d = reducer.fit_transform(emb)

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            emb_2d[:, 0],
            emb_2d[:, 1],
            c=self._cached_labels if hasattr(self, "_cached_labels") else None,
            cmap="Spectral",
            s=5,
            alpha=0.7,
        )
        ax.set_title("UMAP Embedding Colored by Condition")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        if hasattr(self, "_cached_labels"):
            plt.colorbar(scatter, ax=ax, label="Condition")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(save_dir / "benchmark_embedding.png", dpi=300)
        plt.close(fig)
        logger.info("Saved benchmark_embedding.png")

    def cache_embeddings(
        self,
        embeddings: np.ndarray,
        labels: list[int] | None = None,
    ) -> None:
        """Pre-cache embeddings for downstream UMAP plotting.

        Parameters
        ----------
        embeddings : np.ndarray
            ``(n_cells, n_features)`` embedding matrix.
        labels : list[int], optional
            Condition labels for color-coding.
        """
        self._cached_embeddings = embeddings
        self._cached_labels = labels

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_embeddings(
        model: torch.nn.Module,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Extract embedding vectors from the model's penultimate layer.

        Falls back to the model output if a dedicated ``encode`` or
        ``get_embedding`` method is not available.
        """
        if hasattr(model, "encode") and callable(model.encode):
            # scGPT-style encode method
            return model.encode(inputs)  # type: ignore[no-any-unimported]

        if hasattr(model, "get_embedding") and callable(model.get_embedding):
            # Geneformer-style embedding extractor
            return model.get_embedding(inputs)  # type: ignore[no-any-unimported]

        # Default: forward pass, use output
        with torch.no_grad():
            output = model(inputs)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if output.ndim == 2:
                # Classification logits — already a vector per cell
                return output
            # Flatten remaining dimensions
            return output.view(output.size(0), -1)

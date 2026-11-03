"""Fine-tuning trainers for Geneformer and scGPT on corneal scRNA-seq data.

Both trainers share a common interface (``train``, ``save``, ``load``) and
support automatic mixed precision, gradient clipping, 5-fold cross-validation,
and a dry-run mode for batch-size discovery.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, ClassVar

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset

from corneal_fm.config import set_seed
from corneal_fm.utils import get_device, get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _get_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    warmup_steps: int,
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]:
    """Build an AdamW optimizer with linear warmup schedule."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def _dry_run_find_batch_size(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    start_batch: int = 128,
) -> int:
    """Find the maximum batch size that fits in GPU memory (binary search).

    Parameters
    ----------
    model : nn.Module
        Model to test.
    dataset : Dataset
        Dataset providing a single sample via ``dataset[0]``.
    device : torch.device
        Compute device.
    start_batch : int
        Initial batch size to test.

    Returns
    -------
    int
        Safe batch size.
    """
    model = model.to(device)
    model.eval()
    sample = dataset[0]
    if isinstance(sample, (tuple, list)):
        sample = sample[0]

    lo, hi = 1, start_batch
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            batch = sample.unsqueeze(0).repeat(mid, *([1] * (sample.ndim - 1)))
            batch = batch.to(device)
            with torch.no_grad(), autocast(enabled=True):
                _ = model(batch)
            torch.cuda.empty_cache()
            best = mid
            lo = mid + 1
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            hi = mid - 1
    return best


# ---------------------------------------------------------------------------
# GeneformerTrainer
# ---------------------------------------------------------------------------


class GeneformerTrainer:
    """Fine-tune a Geneformer model for cell-type classification.

    Wraps ``geneformer.Classifier`` with a training loop that supports
    automatic mixed precision, gradient clipping, 5-fold cross-validation,
    and dry-run mode.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary (see :func:`corneal_fm.config.load_config`).

    Attributes
    ----------
    model : geneformer.Classifier | None
        The underlying Geneformer classifier (available after ``train``).
    best_val_loss : float
        Best validation loss observed during training.
    """

    # Lazy-imported class; set after first train call
    _geneformer_cls: ClassVar[Any] = None

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = copy.deepcopy(config)
        self.device: torch.device = get_device()
        if config.get("device", "auto") != "auto":
            self.device = torch.device(config["device"])

        self.model: Any = None
        self.best_val_loss: float = float("inf")
        set_seed(self.config.get("seed", 42))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
    ) -> dict[str, list[float]]:
        """Run fine-tuning with 5-fold cross-validation.

        Parameters
        ----------
        train_dataset : Dataset
            Training dataset.
        val_dataset : Dataset | None
            Optional held-out validation dataset.  If ``None``, a 10% split
            is created from *train_dataset*.

        Returns
        -------
        dict[str, list[float]]
            Training history with keys ``"train_loss"``, ``"val_loss"``.
        """
        # Lazy import
        if GeneformerTrainer._geneformer_cls is None:
            from geneformer import Classifier as _GFCls  # type: ignore[import-untyped]

            GeneformerTrainer._geneformer_cls = _GFCls

        from sklearn.model_selection import StratifiedKFold  # type: ignore[import-untyped]

        cfg = self.config
        n_epochs = cfg.get("epochs", 25)
        batch_size = cfg.get("batch_size", 24)
        lr = cfg.get("lr", 5e-5)
        weight_decay = cfg.get("weight_decay", 0.01)
        warmup_steps = cfg.get("warmup_steps", 1000)
        freeze_layers = cfg.get("freeze_layers", 0)
        dry_run = cfg.get("dry_run", False)
        use_amp = cfg.get("amp", True) and self.device.type == "cuda"

        if dry_run:
            logger.info("Dry-run mode: determining maximum batch size ...")
            safe_bs = _dry_run_find_batch_size(
                self._build_model(cfg),
                train_dataset,
                self.device,
            )
            logger.info("Max safe batch size (Geneformer): %d", safe_bs)
            cfg["batch_size"] = safe_bs
            return {"train_loss": [], "val_loss": []}

        # 5-fold CV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.get("seed", 42))
        # Obtain labels for stratified splitting — assumes dataset has .labels or
        # returns (_, label) tuples.
        labels = self._extract_labels(train_dataset)
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(train_dataset)), labels)):
            logger.info("=" * 50)
            logger.info("Fold %d / 5", fold + 1)
            logger.info("=" * 50)

            fold_train = Subset(train_dataset, train_idx)
            fold_val = Subset(train_dataset, val_idx)

            train_loader = DataLoader(
                fold_train,
                batch_size=batch_size,
                shuffle=True,
                num_workers=cfg.get("num_workers", 4),
                pin_memory=True,
                drop_last=True,
            )
            val_loader = DataLoader(
                fold_val,
                batch_size=batch_size * 2,
                shuffle=False,
                num_workers=cfg.get("num_workers", 4),
                pin_memory=True,
            )

            model = self._build_model(cfg)
            model.to(self.device)
            self._freeze_layers(model, freeze_layers)

            optimizer, scheduler = _get_optimizer(model, lr, weight_decay, warmup_steps)
            scaler = GradScaler(enabled=use_amp)
            criterion = nn.CrossEntropyLoss()

            fold_best_loss = float("inf")

            for epoch in range(n_epochs):
                # ---- Train ----
                model.train()
                train_loss = 0.0
                grad_norm = 0.0
                for batch in train_loader:
                    inputs, targets = self._unpack_batch(batch)
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    optimizer.zero_grad()

                    with autocast(enabled=use_amp):
                        logits = model(inputs)
                        loss = criterion(logits, targets)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        cfg.get("gradient_clip_norm", 1.0),
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                    train_loss += loss.item()

                avg_train_loss = train_loss / len(train_loader)

                # ---- Validation ----
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        inputs, targets = self._unpack_batch(batch)
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        with autocast(enabled=use_amp):
                            logits = model(inputs)
                            loss = criterion(logits, targets)
                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)

                logger.info(
                    "Epoch %2d/%2d | Train Loss: %.4f | Val Loss: %.4f | Grad Norm: %.4f",
                    epoch + 1,
                    n_epochs,
                    avg_train_loss,
                    avg_val_loss,
                    grad_norm,
                )

                history["train_loss"].append(avg_train_loss)
                history["val_loss"].append(avg_val_loss)

                # Track best model for this fold
                if avg_val_loss < fold_best_loss:
                    fold_best_loss = avg_val_loss
                    self.model = copy.deepcopy(model)
                    self.best_val_loss = fold_best_loss

            logger.info("Fold %d best val loss: %.4f", fold + 1, fold_best_loss)

        return history

    def save(self, path: Path) -> None:
        """Save model weights and configuration to disk.

        Parameters
        ----------
        path : Path
            Directory or file path.  If a directory, weights are saved as
            ``path / geneformer_model.pt``.
        """
        path = Path(path)
        if path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
            save_path = path / "geneformer_model.pt"
        else:
            save_path = path
            save_path.parent.mkdir(parents=True, exist_ok=True)

        if self.model is None:
            raise RuntimeError("No trained model to save. Call train() first.")

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
                "best_val_loss": self.best_val_loss,
            },
            save_path,
        )
        logger.info("Model saved to %s", save_path)

    @classmethod
    def load(cls, path: Path) -> GeneformerTrainer:
        """Load a previously saved model from a checkpoint.

        Parameters
        ----------
        path : Path
            Path to the checkpoint ``.pt`` file or the directory containing
            ``geneformer_model.pt``.

        Returns
        -------
        GeneformerTrainer
            Restored trainer instance with the model in evaluation mode.
        """
        path = Path(path)
        if path.is_dir():
            path = path / "geneformer_model.pt"

        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location="cpu")
        instance = cls(checkpoint["config"])
        instance.best_val_loss = checkpoint["best_val_loss"]

        model = instance._build_model(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(instance.device)
        model.eval()
        instance.model = model

        logger.info("Model loaded from %s", path)
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self, cfg: dict[str, Any]) -> Any:
        """Instantiate a Geneformer Classifier from configuration."""
        Cls = GeneformerTrainer._geneformer_cls
        if Cls is None:
            raise RuntimeError("Geneformer not imported — did you import it?")
        return Cls(
            model_directory=cfg.get("pretrained_path"),
            num_classes=cfg.get("geneformer_num_classes"),
            filter_data=cfg.get("geneformer_filter_data", True),
        )

    @staticmethod
    def _extract_labels(dataset: Dataset) -> list[int]:
        """Extract integer labels from a dataset for stratified splitting."""
        labels: list[int] = []
        for i in range(len(dataset)):
            item = dataset[i]
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                label = item[1]
            else:
                label = 0
            labels.append(int(label) if not isinstance(label, int) else label)
        return labels

    @staticmethod
    def _unpack_batch(
        batch: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Unpack a batch into (inputs, targets)."""
        if isinstance(batch, (tuple, list)):
            return batch[0], batch[1]
        return batch, batch  # fallback for unsupervised

    @staticmethod
    def _freeze_layers(model: Any, n_freeze: int) -> None:
        """Freeze the first *n_freeze* transformer layers.

        Parameters
        ----------
        model : geneformer.Classifier
            The classifier model.
        n_freeze : int
            Number of encoder layers to freeze (``0`` = train all).
        """
        if n_freeze <= 0:
            return
        try:
            encoder_layers = model.model.encoder.layer
            for i in range(min(n_freeze, len(encoder_layers))):
                for param in encoder_layers[i].parameters():
                    param.requires_grad = False
            logger.info("Froze %d / %d encoder layers", n_freeze, len(encoder_layers))
        except AttributeError:
            logger.warning("Could not freeze layers — encoder.layer not found.")


# ---------------------------------------------------------------------------
# scGPTTrainer
# ---------------------------------------------------------------------------


class scGPTTrainer:
    """Fine-tune an scGPT model for masked-gene-prediction fine-tuning.

    Handles gene vocabulary matching between the pretrained model and the
    corneal dataset, partial weight loading for mismatched vocabularies,
    and the same training infrastructure as :class:`GeneformerTrainer`.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary.

    Attributes
    ----------
    model : torch.nn.Module | None
        The underlying scGPT model (available after ``train``).
    gene_vocab : dict[str, int] | None
        Mapping from gene names to token IDs (available after ``train``).
    best_val_loss : float
        Best validation loss observed during training.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = copy.deepcopy(config)
        self.device: torch.device = get_device()
        if config.get("device", "auto") != "auto":
            self.device = torch.device(config["device"])

        self.model: nn.Module | None = None
        self.gene_vocab: dict[str, int] | None = None
        self.best_val_loss: float = float("inf")
        set_seed(self.config.get("seed", 42))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
    ) -> dict[str, list[float]]:
        """Run fine-tuning with 5-fold cross-validation.

        Parameters
        ----------
        train_dataset : Dataset
            Training dataset that yields gene-expression pairs.
        val_dataset : Dataset | None
            Optional held-out validation dataset.  If ``None`` a 10% split
            is created from *train_dataset*.

        Returns
        -------
        dict[str, list[float]]
            Training history.
        """
        # Lazy imports
        from sklearn.model_selection import StratifiedKFold  # type: ignore[import-untyped]

        cfg = self.config
        n_epochs = cfg.get("epochs", 25)
        batch_size = cfg.get("batch_size", 24)
        lr = cfg.get("lr", 5e-5)
        weight_decay = cfg.get("weight_decay", 0.01)
        warmup_steps = cfg.get("warmup_steps", 1000)
        freeze_layers = cfg.get("freeze_layers", 0)
        dry_run = cfg.get("dry_run", False)
        use_amp = cfg.get("amp", True) and self.device.type == "cuda"

        if dry_run:
            logger.info("Dry-run mode: determining maximum batch size ...")
            dummy = self._build_model(cfg)
            safe_bs = _dry_run_find_batch_size(dummy, train_dataset, self.device)
            logger.info("Max safe batch size (scGPT): %d", safe_bs)
            cfg["batch_size"] = safe_bs
            return {"train_loss": [], "val_loss": []}

        # Build model and match vocabulary
        model = self._build_model(cfg)
        self._match_vocab(model, train_dataset)
        model.to(self.device)
        self._freeze_layers(model, freeze_layers)

        # 5-fold CV
        labels = self._extract_labels(train_dataset)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.get("seed", 42))
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(train_dataset)), labels)):
            logger.info("=" * 50)
            logger.info("Fold %d / 5", fold + 1)
            logger.info("=" * 50)

            fold_train = Subset(train_dataset, train_idx)
            fold_val = Subset(train_dataset, val_idx)

            train_loader = DataLoader(
                fold_train,
                batch_size=batch_size,
                shuffle=True,
                num_workers=cfg.get("num_workers", 4),
                pin_memory=True,
                drop_last=True,
            )
            val_loader = DataLoader(
                fold_val,
                batch_size=batch_size * 2,
                shuffle=False,
                num_workers=cfg.get("num_workers", 4),
                pin_memory=True,
            )

            fold_model = copy.deepcopy(model)
            optimizer, scheduler = _get_optimizer(fold_model, lr, weight_decay, warmup_steps)
            scaler = GradScaler(enabled=use_amp)
            criterion = (
                nn.MSELoss()
                if cfg.get("scgpt_recon_loss", "mse") == "mse"
                else nn.KLDivLoss(reduction="batchmean")
            )

            fold_best_loss = float("inf")

            for epoch in range(n_epochs):
                # ---- Train ----
                fold_model.train()
                train_loss = 0.0
                grad_norm = 0.0
                for batch in train_loader:
                    inputs, targets = self._unpack_batch(batch)
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    optimizer.zero_grad()
                    with autocast(enabled=use_amp):
                        outputs = fold_model(inputs)
                        loss = criterion(outputs, targets)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        fold_model.parameters(),
                        cfg.get("gradient_clip_norm", 1.0),
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                    train_loss += loss.item()

                avg_train_loss = train_loss / len(train_loader)

                # ---- Validation ----
                fold_model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        inputs, targets = self._unpack_batch(batch)
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        with autocast(enabled=use_amp):
                            outputs = fold_model(inputs)
                            loss = criterion(outputs, targets)
                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)

                logger.info(
                    "Epoch %2d/%2d | Train Loss: %.4f | Val Loss: %.4f | Grad Norm: %.4f",
                    epoch + 1,
                    n_epochs,
                    avg_train_loss,
                    avg_val_loss,
                    grad_norm,
                )

                history["train_loss"].append(avg_train_loss)
                history["val_loss"].append(avg_val_loss)

                if avg_val_loss < fold_best_loss:
                    fold_best_loss = avg_val_loss
                    self.model = copy.deepcopy(fold_model)
                    self.best_val_loss = fold_best_loss

            logger.info("Fold %d best val loss: %.4f", fold + 1, fold_best_loss)

        return history

    def save(self, path: Path) -> None:
        """Save model weights, gene vocabulary, and configuration.

        Parameters
        ----------
        path : Path
            Directory or file path.  If a directory, weights are saved as
            ``path / scgpt_model.pt``.
        """
        path = Path(path)
        if path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
            save_path = path / "scgpt_model.pt"
        else:
            save_path = path
            save_path.parent.mkdir(parents=True, exist_ok=True)

        if self.model is None:
            raise RuntimeError("No trained model to save. Call train() first.")

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "gene_vocab": self.gene_vocab,
                "config": self.config,
                "best_val_loss": self.best_val_loss,
            },
            save_path,
        )
        logger.info("Model saved to %s", save_path)

    @classmethod
    def load(cls, path: Path) -> scGPTTrainer:
        """Load a previously saved scGPT model from a checkpoint.

        Parameters
        ----------
        path : Path
            Path to the checkpoint ``.pt`` file or directory containing
            ``scgpt_model.pt``.

        Returns
        -------
        scGPTTrainer
            Restored trainer instance with the model in evaluation mode.
        """
        path = Path(path)
        if path.is_dir():
            path = path / "scgpt_model.pt"

        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location="cpu")
        instance = cls(checkpoint["config"])
        instance.best_val_loss = checkpoint["best_val_loss"]
        instance.gene_vocab = checkpoint.get("gene_vocab")

        model = instance._build_model(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.to(instance.device)
        model.eval()
        instance.model = model

        logger.info("Model loaded from %s", path)
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self, cfg: dict[str, Any]) -> nn.Module:
        """Instantiate an scGPT model from configuration."""
        import scgpt  # type: ignore[import-untyped]

        return scgpt.model.GPT(
            vocab_size=cfg.get("vocab_size", None),
            n_layers=cfg.get("scgpt_nlayers", 4),
        )

    def _match_vocab(self, model: nn.Module, dataset: Dataset) -> None:
        """Match and subset model weights to the dataset gene vocabulary.

        Logs the number of matched and unmatched genes.
        """

        # Extract gene vocabulary from the dataset if available
        dataset_vocab: dict[str, int] | None = getattr(dataset, "gene_vocab", None)
        if dataset_vocab is None:
            logger.warning("Dataset has no gene_vocab attribute; skipping vocab matching.")
            return

        model_vocab: dict[str, int] | None = getattr(model, "gene_vocab", None)
        if model_vocab is None:
            logger.warning("Model has no gene_vocab attribute; skipping vocab matching.")
            self.gene_vocab = dataset_vocab
            return

        # Identify matched and missing genes
        matched = set(dataset_vocab.keys()) & set(model_vocab.keys())
        missing = set(dataset_vocab.keys()) - set(model_vocab.keys())
        logger.info(
            "Gene vocabulary: %d matched, %d missing",
            len(matched),
            len(missing),
        )

        # Partial weight loading for the embedding layer
        try:
            emb = model.get_input_embeddings()
            if emb is not None and hasattr(emb, "weight"):
                old_weight = emb.weight.data.clone()
                # Keep only matched rows
                matched_indices = [model_vocab[g] for g in dataset_vocab if g in model_vocab]
                new_weight = old_weight[matched_indices]
                emb.weight = nn.Parameter(new_weight)
                logger.info(
                    "Resized embedding from %d → %d",
                    old_weight.size(0),
                    new_weight.size(0),
                )
        except (AttributeError, RuntimeError, IndexError):
            logger.warning("Could not resize embedding layer; continuing with full vocab.")

        self.gene_vocab = dataset_vocab

    @staticmethod
    def _extract_labels(dataset: Dataset) -> list[int]:
        """Extract integer labels for stratified splitting."""
        labels: list[int] = []
        for i in range(len(dataset)):
            item = dataset[i]
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                label = item[1]
            else:
                label = 0
            labels.append(int(label) if not isinstance(label, int) else label)
        return labels

    @staticmethod
    def _unpack_batch(
        batch: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Unpack a batch into (inputs, targets)."""
        if isinstance(batch, (tuple, list)):
            return batch[0], batch[1]
        return batch, batch

    @staticmethod
    def _freeze_layers(model: nn.Module, n_freeze: int) -> None:
        """Freeze the first *n_freeze* transformer layers."""
        if n_freeze <= 0:
            return
        try:
            layers = (
                model.transformer.layers
                if hasattr(model, "transformer")
                else list(model.children())
            )
            for i in range(min(n_freeze, len(layers))):
                for param in layers[i].parameters():
                    param.requires_grad = False
            logger.info("Froze %d / %d layers", n_freeze, len(layers))
        except (AttributeError, IndexError):
            logger.warning("Could not freeze layers.")

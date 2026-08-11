"""Compatibility shim for Geneformer tokenization on modern pandas (>= 2.0).

Geneformer's in-memory h5ad tokenizer ``TranscriptomeTokenizer.tokenize_anndata``
indexes ``adata.var["ensembl_id_collapsed"]`` with a *positional* integer array
via pandas ``.loc`` (tokenizer.py: ``...["ensembl_id_collapsed"][coding_miRNA_loc]``).
``sum_ensembl_ids`` first sets ``var_names`` to the collapsed Ensembl IDs — a
*string* index — so on pandas >= 2.0 the integer array is treated as label-based
(not positional) and raises ``KeyError``. The loom path is unaffected because it
uses loompy's positional ``.ra`` accessor.

This shim wraps ``geneformer.tokenizer.sum_ensembl_ids`` so the returned AnnData
carries a positional ``var`` RangeIndex, restoring the positional indexing the
tokenizer relies on. The Ensembl identifiers are preserved in the
``ensembl_id_collapsed`` column, which is exactly what the downstream code reads.
"""

from __future__ import annotations

import geneformer.tokenizer as _gftok

_orig_sum_ensembl_ids = _gftok.sum_ensembl_ids
_installed = False


def install() -> None:
    """Patch sum_ensembl_ids once so tokenize_anndata works on pandas >= 2.0."""
    global _installed
    if _installed:
        return

    def _sum_with_positional_var_index(*args, **kwargs):
        out = _orig_sum_ensembl_ids(*args, **kwargs)
        try:
            out.var.reset_index(drop=True, inplace=True)
        except Exception:
            pass
        return out

    _gftok.sum_ensembl_ids = _sum_with_positional_var_index
    _installed = True

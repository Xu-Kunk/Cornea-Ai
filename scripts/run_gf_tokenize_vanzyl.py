#!/usr/bin/env python
"""Run protocol_04_tokenize_gf.py on <compute-node> with the pandas>=2 var-index shim.

Environment-completion wrapper (no science changes):
  - geneformer 0.1.0's ``sum_ensembl_ids`` returns an AnnData whose ``var``
    index holds the collapsed Ensembl IDs (strings); ``tokenize_anndata`` then
    indexes ``adata.var["ensembl_id_collapsed"]`` with a *positional* integer
    array, which pandas >= 2.0 treats as label-based and raises KeyError.
  - The shim below resets the returned var index to a positional RangeIndex
    (Ensembl IDs stay in the ``ensembl_id_collapsed`` column — exactly what the
    tokenizer reads downstream). Logic is a verbatim copy of
    corneal_fm/gf_compat.py (the project's established fix for the same issue
    on the local GSE153515 chain).

Usage (all args are forwarded to protocol_04):
    python run_gf_tokenize_vanzyl.py --input data/vanzyl_annotated.h5ad \
        --output-dir data --prefix gf_tokenized_vanzyl \
        --custom-attrs cell_type,disease,tissue,donor_id,author_cell_type,cell_class_4
"""

from __future__ import annotations

import runpy
import sys

import geneformer.tokenizer as _gftok

_orig_sum_ensembl_ids = _gftok.sum_ensembl_ids


def _sum_with_positional_var_index(*args, **kwargs):
    out = _orig_sum_ensembl_ids(*args, **kwargs)
    try:
        out.var.reset_index(drop=True, inplace=True)
    except Exception:
        pass
    return out


_gftok.sum_ensembl_ids = _sum_with_positional_var_index

sys.argv = ["protocol_04_tokenize_gf.py"] + sys.argv[1:]
runpy.run_path(
    "<home>/corneal_scgpt/scripts/protocol_04_tokenize_gf.py",
    run_name="__main__",
)

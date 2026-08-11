"""Command-line entry point for GPR measurement and paired comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from corneal_fm.gpr_evaluation import (
    compare_checkpoints,
    evaluate_checkpoint,
    evaluate_embeddings,
)
from corneal_fm.gpr_schema import ComparisonRequest, ComparisonResult, SingleResult


def _write_result(path: str, result: ComparisonResult | SingleResult) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPR / Continuity-Score measurement (Methods §4.4)",
    )
    parser.add_argument("--checkpoint", default=None, help="Fine-tuned checkpoint dir")
    parser.add_argument("--before_checkpoint", default=None, help="Pre-fine-tune checkpoint")
    parser.add_argument("--after_checkpoint", default=None, help="Post-fine-tune checkpoint")
    parser.add_argument("--tokenized", default=None, help="labeled.dataset with layer column")
    parser.add_argument("--embeddings", default=None, help="Precomputed [N,D] .npy")
    parser.add_argument("--layers", default=None, help="Per-cell layer-id .npy")
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument(
        "--cs_before",
        type=float,
        default=None,
        help="Legacy bare scalar (rejected; use paired checkpoint mode)",
    )
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_mean_pool", action="store_true")
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()
    if args.cs_before is not None:
        parser.error("--cs_before bare scalar injection is prohibited; use paired checkpoint mode")

    compare_requested = bool(args.before_checkpoint or args.after_checkpoint)
    if compare_requested:
        if not (args.before_checkpoint and args.after_checkpoint and args.tokenized):
            parser.error("Paired mode requires --before_checkpoint --after_checkpoint --tokenized")
        if args.use_mean_pool:
            parser.error("Paired mode fixes raw CLS as primary and mean pooling as sensitivity")
        comparison = compare_checkpoints(
            ComparisonRequest(
                before_checkpoint=args.before_checkpoint,
                after_checkpoint=args.after_checkpoint,
                tokenized_path=args.tokenized,
                num_classes=args.num_classes,
                seed=args.seed,
                n_boot=args.n_boot,
            ),
        )
        output = args.out or "gpr_comparison_result.json"
        _write_result(output, comparison)
        primary = comparison["primary"]
        print("=" * 60)
        print(f"Primary raw CLS CS_before: {primary['cs_before']}")
        print(f"Primary raw CLS CS_after:  {primary['cs_after']}")
        print(f"GPR:                       {primary['gpr']}")
        print(f"Absolute delta:            {primary['absolute_delta']}")
        print(f"saved -> {output}")
        print("=" * 60)
        return

    if args.embeddings and args.layers:
        result = evaluate_embeddings(args.embeddings, args.layers, n_boot=args.n_boot)
    elif args.checkpoint and args.tokenized:
        result = evaluate_checkpoint(
            args.checkpoint,
            args.tokenized,
            num_classes=args.num_classes,
            n_boot=args.n_boot,
            use_cls=not args.use_mean_pool,
            seed=args.seed,
        )
    else:
        parser.error(
            "Provide paired checkpoints, (--embeddings --layers), or (--checkpoint --tokenized)",
        )
    output = args.out or "cs_gpr_result.json"
    _write_result(output, result)
    print("=" * 60)
    print(f"Continuity Score:  {result['continuity_score']}")
    print(f"  pair similarity: {result['pair_similarity']}")
    print(f"  layers present:  {result['layers_present']}")
    print(f"  LAA:             {result['layer_adjacency_accuracy']}")
    print(f"  bootstrap 95%CI: {result['bootstrap_cs_95ci']}")
    print(f"saved -> {output}")
    print("=" * 60)


if __name__ == "__main__":
    main()

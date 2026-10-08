#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from activation_drift.evaluation import evaluate_conversation_quality
from activation_drift.process_results import load_raw_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate conversation quality from a saved probe-drift result directory."
    )
    parser.add_argument(
        "--result-dir",
        required=True,
        help="Result directory containing runs.jsonl and probe_scores_by_turn.csv",
    )
    parser.add_argument(
        "--dump-file",
        required=True,
        help="Output pickle path for all evaluation results",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name, to choose between qwen and llama8b.",
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parent),
        help="Repository root used to resolve the default normalization path",
    )
    parser.add_argument(
        "--normalization-json",
        default=None,
        help="Override normalization JSON path. Defaults to {repo_root}/probes/{model}/best_probe_calibration_train.json",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum number of assistant turns to evaluate per run",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-oss:120b-cloud",
        help="Ollama judge model used for coherence evaluation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Judge sampling temperature for coherence evaluation",
    )
    parser.add_argument(
        "--repetition-threshold",
        type=float,
        default=0.8,
        help="Cosine similarity threshold for sentence repetition",
    )
    parser.add_argument(
        "--repetition-model-name",
        default="all-mpnet-base-v2",
        help="SentenceTransformer model for repetition scoring",
    )
    parser.add_argument(
        "--coherence-only",
        action="store_true",
        help="Run only coherence evaluation",
    )
    parser.add_argument(
        "--repetition-only",
        action="store_true",
        help="Run only repetition evaluation",
    )
    parser.add_argument(
        "--remove-reasoning-trail",
        action="store_true",
        help="Remove tagged <think> reasoning before evaluating responses",
    )
    return parser


def _resolve_normalization_json(repo_root: Path, model: str, override: str | None) -> Path:
    if override:
        return Path(override)
    return repo_root / f"probes/{model}/best_probe_calibration_train.json"


def main() -> int:
    args = build_parser().parse_args()

    if args.coherence_only and args.repetition_only:
        raise ValueError("--coherence-only and --repetition-only are mutually exclusive")

    measure_coherence = not args.repetition_only
    measure_repetition = not args.coherence_only

    repo_root = Path(args.repo_root).expanduser().resolve()
    result_dir = Path(args.result_dir).expanduser().resolve()
    dump_file = Path(args.dump_file).expanduser().resolve()
    normalization_json = _resolve_normalization_json(
        repo_root=repo_root,
        model=str(args.model),
        override=args.normalization_json,
    ).expanduser().resolve()

    print(f"Repo root          : {repo_root}")
    print(f"Result dir         : {result_dir}")
    print(f"Normalization json : {normalization_json}")
    print(f"Dump file          : {dump_file}")
    print(f"Measure coherence  : {measure_coherence}")
    print(f"Measure repetition : {measure_repetition}")
    print(f"Remove reasoning   : {args.remove_reasoning_trail}")

    probe_scores_by_turn_df, normalization_stats, run_records = load_raw_results(
        result_dir,
        normalization_json=normalization_json,
    )

    print(f"Loaded run records : {len(run_records)}")
    print(f"Probe rows         : {len(probe_scores_by_turn_df)}")
    print(f"Norm values        : {len(normalization_stats)}")

    results = evaluate_conversation_quality(
        run_records,
        measure_coherence=measure_coherence,
        measure_repetition=measure_repetition,
        judge_model=args.judge_model,
        max_turns=int(args.max_turns),
        temperature=float(args.temperature),
        repetition_threshold=float(args.repetition_threshold),
        repetition_model_name=str(args.repetition_model_name),
        remove_reasoning_trail=bool(args.remove_reasoning_trail),
        dump_file=dump_file,
    )

    print("\nSummary")
    for metric_name, metric_df in results.items():
        print(f"\n[{metric_name}]")
        print(metric_df)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
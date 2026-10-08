#!/usr/bin/env python3
"""Unified Schwartz dataset activation processor.

Supports:
- ValueNet -> binary positive/negative per value
- ValueEval -> Schwartz-10 binary labels (max/OR mapping)
- FULCRA -> Schwartz-10 continuous labels with masks

Examples:
  python process_schwartz.py valuenet --dataset-dir data/ValueNet/v0.3_original
  python process_schwartz.py valueeval --dataset-dir data/touche23_valueeval
  python process_schwartz.py fulcra --dataset-dir data/value_fulcra
  python process_schwartz.py all
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import pandas as pd

from activation_drift import BehaviorActivationProcessor
from activation_drift.utils import DEFAULT_LLM_MODEL_PATH


DEFAULT_MODEL_PATH = DEFAULT_LLM_MODEL_PATH

SCHWARTZ_10_VALUES = [
    "SELF-DIRECTION",
    "STIMULATION",
    "HEDONISM",
    "ACHIEVEMENT",
    "POWER",
    "SECURITY",
    "TRADITION",
    "CONFORMITY",
    "BENEVOLENCE",
    "UNIVERSALISM",
]

VALUEEVAL_TO_SCHWARTZ10 = {
    "SELF-DIRECTION": ["Self-direction: thought", "Self-direction: action"],
    "STIMULATION": ["Stimulation"],
    "HEDONISM": ["Hedonism"],
    "ACHIEVEMENT": ["Achievement"],
    "POWER": ["Power: dominance", "Power: resources", "Face"],
    "SECURITY": ["Security: personal", "Security: societal"],
    "TRADITION": ["Tradition"],
    "CONFORMITY": ["Conformity: rules", "Conformity: interpersonal", "Humility"],
    "BENEVOLENCE": ["Benevolence: caring", "Benevolence: dependability"],
    "UNIVERSALISM": [
        "Universalism: concern",
        "Universalism: nature",
        "Universalism: tolerance",
        "Universalism: objectivity",
    ],
}

FULCRA_TYPE_TO_SCHWARTZ10 = {
    "self-direction": "SELF-DIRECTION",
    "stimulation": "STIMULATION",
    "hedonism": "HEDONISM",
    "achievement": "ACHIEVEMENT",
    "power": "POWER",
    "security": "SECURITY",
    "tradition": "TRADITION",
    "conformity": "CONFORMITY",
    "benevolence": "BENEVOLENCE",
    "universalism": "UNIVERSALISM",
}


def _labels_to_schwartz10_valueeval(row: pd.Series) -> list[float]:
    target = []
    for value_name in SCHWARTZ_10_VALUES:
        cols = VALUEEVAL_TO_SCHWARTZ10[value_name]
        vals = [float(row[c]) for c in cols]
        target.append(float(max(vals)))
    return target


def _discover_valuenet_csvs(dataset_dir: Path):
    excluded = {"train.csv", "test.csv", "eval.csv", "meta.csv"}
    return sorted(p for p in dataset_dir.glob("*.csv") if p.name not in excluded)


def _label_to_polarity(label) -> str:
    val = int(label)
    if val == 1:
        return "positive"
    if val in (0, -1):
        return "negative"
    raise ValueError(f"Unexpected label '{label}'. Expected one of -1, 0, 1.")


def run_valuenet(args) -> int:
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"Error: dataset directory not found: {dataset_dir}")
        return 1

    value_csvs = _discover_valuenet_csvs(dataset_dir)
    if not value_csvs:
        print(f"Error: no value CSVs found in {dataset_dir}")
        return 1

    print("=" * 80)
    print("ValueNet Schwartz Activation Processor")
    print("=" * 80)
    print(f"Dataset dir : {dataset_dir}")
    print(f"Model       : {args.model_path}")
    print(f"Text column : {args.text_column}")
    print(f"Label column: {args.label_column}")
    print(f"Output dir  : {args.output_dir}")
    print(f"Max length  : {args.max_length}")
    if args.limit_per_class is not None:
        print(f"Limit/class : {args.limit_per_class}")
    print(f"Values found: {len(value_csvs)}")
    print("=" * 80)

    processor = BehaviorActivationProcessor(args.model_path, quantization=args.quantization, output_dir=args.output_dir)
    overall_counts = {}

    for csv_path in value_csvs:
        value_name = csv_path.stem
        print(f"\nProcessing value: {value_name} ({csv_path.name})")

        df = pd.read_csv(csv_path)
        if args.text_column not in df.columns:
            raise ValueError(f"Missing text column '{args.text_column}' in {csv_path}")
        if args.label_column not in df.columns:
            raise ValueError(f"Missing label column '{args.label_column}' in {csv_path}")

        counts = {"positive": 0, "negative": 0, "skipped": 0}

        for idx, row in df.iterrows():
            text = str(row[args.text_column]).strip()
            if not text:
                counts["skipped"] += 1
                continue

            polarity = _label_to_polarity(row[args.label_column])
            behavior_type = f"{value_name}_{polarity}"

            if args.limit_per_class is not None and counts[polarity] >= args.limit_per_class:
                continue

            example_id = f"{value_name}_{polarity}_{idx}"
            ok = processor._process_single_example(
                text=text,
                example_id=example_id,
                behavior_type=behavior_type,
                max_length=args.max_length,
            )
            if ok:
                counts[polarity] += 1
            else:
                counts["skipped"] += 1

        overall_counts[value_name] = counts
        print(
            f"✓ {value_name}: +{counts['positive']} / -{counts['negative']} / skipped={counts['skipped']}"
        )

    print("\n" + "=" * 80)
    print("Processing Complete")
    print("=" * 80)

    total_pos = total_neg = total_skipped = 0
    for value_name, counts in overall_counts.items():
        total_pos += counts["positive"]
        total_neg += counts["negative"]
        total_skipped += counts["skipped"]
        print(
            f"{value_name:<15} positive={counts['positive']:<5} "
            f"negative={counts['negative']:<5} skipped={counts['skipped']}"
        )

    print("-" * 80)
    print(f"TOTAL positive={total_pos} negative={total_neg} skipped={total_skipped}")
    print(f"Saved under: ./{args.output_dir}/")
    return 0


def _find_valueeval_label_files(dataset_dir: Path):
    return sorted(p for p in dataset_dir.glob("labels-*.tsv") if not p.name.startswith("level1-"))


def _valueeval_args_file(label_file: Path) -> Path:
    suffix = label_file.name[len("labels-"):]
    return label_file.parent / f"arguments-{suffix}"


def run_valueeval(args) -> int:
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"Error: dataset directory not found: {dataset_dir}")
        return 1

    label_files = _find_valueeval_label_files(dataset_dir)
    if not label_files:
        print(f"Error: no ValueEval label files found in {dataset_dir}")
        return 1

    print("=" * 80)
    print("ValueEval Activation Processor")
    print("=" * 80)
    print(f"Dataset dir : {dataset_dir}")
    print(f"Model       : {args.model_path}")
    print(f"Output dir  : {args.output_dir}")
    print(f"Max length  : {args.max_length}")
    if args.limit_per_split is not None:
        print(f"Limit/split : {args.limit_per_split}")
    print(f"Label files : {len(label_files)}")
    print("=" * 80)

    processor = BehaviorActivationProcessor(args.model_path, quantization=args.quantization, output_dir=args.output_dir)

    total_saved = 0
    for label_file in label_files:
        arguments_file = _valueeval_args_file(label_file)
        if not arguments_file.exists():
            print(f"Skipping {label_file.name}: missing {arguments_file.name}")
            continue

        split_name = label_file.stem[len("labels-"):]
        print(f"\nProcessing split: {split_name}")

        labels_df = pd.read_csv(label_file, sep="\t")
        args_df = pd.read_csv(arguments_file, sep="\t")

        required_cols = ["Argument ID", "Conclusion", "Stance", "Premise"]
        for col in required_cols:
            if col not in args_df.columns:
                raise ValueError(f"Missing column '{col}' in {arguments_file}")

        missing_label_cols = []
        for cols in VALUEEVAL_TO_SCHWARTZ10.values():
            for col in cols:
                if col not in labels_df.columns:
                    missing_label_cols.append(col)
        if missing_label_cols:
            raise ValueError(
                f"Missing label columns in {label_file}: {sorted(set(missing_label_cols))}"
            )

        merged = args_df.merge(labels_df, on="Argument ID", how="inner")
        if merged.empty:
            print(f"  Warning: no merged examples for split {split_name}")
            continue

        saved_in_split = 0
        for idx, row in merged.iterrows():
            if args.limit_per_split is not None and saved_in_split >= args.limit_per_split:
                break

            arg_id = str(row["Argument ID"])
            text = (
                f"Conclusion: {str(row['Conclusion']).strip()}\n"
                f"Stance: {str(row['Stance']).strip()}\n"
                f"Premise: {str(row['Premise']).strip()}"
            )

            target = _labels_to_schwartz10_valueeval(row)
            mask = [1.0] * len(SCHWARTZ_10_VALUES)

            extra_metadata = {
                "dataset": "valueeval",
                "split": split_name,
                "source_file": label_file.name,
                "argument_id": arg_id,
                "schwartz_10_binary": target,
                "schwartz_10_binary_mask": mask,
                "schwartz_10_values": SCHWARTZ_10_VALUES,
            }

            behavior_type = f"valueeval_{split_name}"
            example_id = f"valueeval_{split_name}_{arg_id}_{idx}"
            ok = processor._process_single_example(
                text=text,
                example_id=example_id,
                behavior_type=behavior_type,
                max_length=args.max_length,
                extra_metadata=extra_metadata,
            )
            if ok:
                saved_in_split += 1

        total_saved += saved_in_split
        print(f"  Saved {saved_in_split} examples")

    print("\n" + "=" * 80)
    print("Processing complete")
    print("=" * 80)
    print(f"Total saved: {total_saved}")
    print(f"Saved under: ./{args.output_dir}/")
    return 0


def _parse_fulcra_value_type_entry(entry: str):
    text = str(entry).strip().lower()
    match = re.match(r"^\s*([^:]+)\s*:\s*([+-]?\d+(?:\.\d+)?)\s*$", text)
    if not match:
        return None
    return match.group(1).strip(), float(match.group(2))


def _build_fulcra_continuous_target(sample: dict):
    per_value_scores = {k: [] for k in SCHWARTZ_10_VALUES}
    for entry in sample.get("value_types", []) or []:
        parsed = _parse_fulcra_value_type_entry(entry)
        if parsed is None:
            continue
        raw_type, raw_score = parsed
        mapped = FULCRA_TYPE_TO_SCHWARTZ10.get(raw_type)
        if mapped is None:
            continue
        clipped = max(-1.0, min(1.0, raw_score))
        per_value_scores[mapped].append(clipped)

    target = [0.0] * len(SCHWARTZ_10_VALUES)
    mask = [0.0] * len(SCHWARTZ_10_VALUES)
    for i, value_name in enumerate(SCHWARTZ_10_VALUES):
        scores = per_value_scores[value_name]
        if not scores:
            continue
        avg = sum(scores) / len(scores)
        target[i] = (avg + 1.0) / 2.0
        mask[i] = 1.0
    return target, mask


def run_fulcra(args) -> int:
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"Error: dataset directory not found: {dataset_dir}")
        return 1

    jsonl_files = sorted(dataset_dir.glob("*.jsonl"))
    if args.files:
        wanted = set(args.files)
        jsonl_files = [p for p in jsonl_files if p.name in wanted]

    if not jsonl_files:
        print(f"Error: no JSONL files to process in {dataset_dir}")
        return 1

    print("=" * 80)
    print("FULCRA Activation Processor")
    print("=" * 80)
    print(f"Dataset dir : {dataset_dir}")
    print(f"Model       : {args.model_path}")
    print(f"Output dir  : {args.output_dir}")
    print(f"Max length  : {args.max_length}")
    if args.limit_per_file is not None:
        print(f"Limit/file  : {args.limit_per_file}")
    print(f"JSONL files : {len(jsonl_files)}")
    print("=" * 80)

    processor = BehaviorActivationProcessor(args.model_path, quantization=args.quantization, output_dir=args.output_dir)
    total_saved = 0
    total_skipped = 0

    for jsonl_path in jsonl_files:
        print(f"\nProcessing file: {jsonl_path.name}")
        saved_in_file = 0
        skipped_in_file = 0

        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for line_idx, line in enumerate(handle):
                if args.limit_per_file is not None and saved_in_file >= args.limit_per_file:
                    break

                line = line.strip()
                if not line:
                    skipped_in_file += 1
                    continue

                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    skipped_in_file += 1
                    continue

                text = str(sample.get("dialogue", "")).strip()
                if not text:
                    skipped_in_file += 1
                    continue

                target, mask = _build_fulcra_continuous_target(sample)
                if sum(mask) == 0:
                    skipped_in_file += 1
                    continue

                extra_metadata = {
                    "dataset": "fulcra",
                    "source_file": jsonl_path.name,
                    "line_index": line_idx,
                    "query_source": sample.get("query_source"),
                    "response_source": sample.get("response_source"),
                    "schwartz_10_continuous": target,
                    "schwartz_10_continuous_mask": mask,
                    "schwartz_10_values": SCHWARTZ_10_VALUES,
                }

                behavior_type = f"fulcra_{jsonl_path.stem}"
                example_id = f"fulcra_{jsonl_path.stem}_{line_idx}"
                ok = processor._process_single_example(
                    text=text,
                    example_id=example_id,
                    behavior_type=behavior_type,
                    max_length=args.max_length,
                    extra_metadata=extra_metadata,
                )
                if ok:
                    saved_in_file += 1
                else:
                    skipped_in_file += 1

        total_saved += saved_in_file
        total_skipped += skipped_in_file
        print(f"  Saved={saved_in_file} | Skipped={skipped_in_file}")

    print("\n" + "=" * 80)
    print("Processing complete")
    print("=" * 80)
    print(f"Total saved   : {total_saved}")
    print(f"Total skipped : {total_skipped}")
    print(f"Saved under   : ./{args.output_dir}/")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified Schwartz activation processor for ValueNet, ValueEval, and FULCRA"
    )
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    valuenet = subparsers.add_parser("valuenet", help="Process ValueNet Schwartz CSVs")
    valuenet.add_argument("--dataset-dir", default="data/ValueNet/v0.3_original")
    valuenet.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    valuenet.add_argument("--quantization", default="none")
    valuenet.add_argument("--text-column", default="scenario")
    valuenet.add_argument("--label-column", default="label")
    valuenet.add_argument("--max-length", type=int, default=512)
    valuenet.add_argument("--output-dir", default="activations/valuenet")
    valuenet.add_argument("--limit-per-class", type=int, default=None)

    valueeval = subparsers.add_parser("valueeval", help="Process ValueEval TSVs")
    valueeval.add_argument("--dataset-dir", default="data/touche23_valueeval")
    valueeval.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    valueeval.add_argument("--quantization", default="none")
    valueeval.add_argument("--max-length", type=int, default=512)
    valueeval.add_argument("--output-dir", default="activations/valueeval")
    valueeval.add_argument("--limit-per-split", type=int, default=None)

    fulcra = subparsers.add_parser("fulcra", help="Process FULCRA JSONL files")
    fulcra.add_argument("--dataset-dir", default="data/value_fulcra")
    fulcra.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    fulcra.add_argument("--quantization", default="none")
    fulcra.add_argument("--max-length", type=int, default=512)
    fulcra.add_argument("--output-dir", default="activations/fulcra")
    fulcra.add_argument("--limit-per-file", type=int, default=None)
    fulcra.add_argument("--files", nargs="*", default=None)

    all_cmd = subparsers.add_parser("all", help="Run ValueNet, ValueEval, and FULCRA sequentially")
    all_cmd.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    all_cmd.add_argument("--quantization", default="none")
    all_cmd.add_argument("--max-length", type=int, default=512)
    all_cmd.add_argument("--valuenet-dir", default="data/ValueNet/v0.3_original")
    all_cmd.add_argument("--valueeval-dir", default="data/touche23_valueeval")
    all_cmd.add_argument("--fulcra-dir", default="data/value_fulcra")
    all_cmd.add_argument("--valuenet-output", default="activations/valuenet")
    all_cmd.add_argument("--valueeval-output", default="activations/valueeval")
    all_cmd.add_argument("--fulcra-output", default="activations/fulcra")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.dataset == "valuenet":
        return run_valuenet(args)
    if args.dataset == "valueeval":
        return run_valueeval(args)
    if args.dataset == "fulcra":
        return run_fulcra(args)
    if args.dataset == "all":
        print("Running all Schwartz processors sequentially")

        a1 = argparse.Namespace(
            dataset_dir=args.valuenet_dir,
            model_path=args.model_path,
            quantization=args.quantization,
            text_column="scenario",
            label_column="label",
            max_length=args.max_length,
            output_dir=args.valuenet_output,
            limit_per_class=None,
        )
        rc = run_valuenet(a1)
        if rc != 0:
            return rc

        a2 = argparse.Namespace(
            dataset_dir=args.valueeval_dir,
            model_path=args.model_path,
            quantization=args.quantization,
            max_length=args.max_length,
            output_dir=args.valueeval_output,
            limit_per_split=None,
        )
        rc = run_valueeval(a2)
        if rc != 0:
            return rc

        a3 = argparse.Namespace(
            dataset_dir=args.fulcra_dir,
            model_path=args.model_path,
            quantization=args.quantization,
            max_length=args.max_length,
            output_dir=args.fulcra_output,
            limit_per_file=None,
            files=None,
        )
        return run_fulcra(a3)

    parser.error(f"Unknown dataset command: {args.dataset}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

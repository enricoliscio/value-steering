#!/usr/bin/env python3
"""Evaluate a trained Schwartz probe on held-out activations.

This script validates a saved probe checkpoint on activation folders produced by
ValueNet, ValueEval, and/or FULCRA processing.

It reports:
- Binary metrics: AUROC, F1, accuracy
- Continuous metrics: MAE, Spearman correlation
- Per-dataset and overall summaries

Example:
  python -m activation_drift.cli.eval_schwartz_probe \
    --checkpoint artifacts/schwartz_probe_valuenet_valueeval_l24.pt \
    --valuenet-activations activations/valuenet \
    --valueeval-activations activations/valueeval \
    --val-ratio 0.1
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from activation_drift.schwartz_probe import (
    SCHWARTZ_10_VALUES,
    LinearSchwartzProbe,
    _load_hidden_vector,
    _safe_relpath,
    _split_train_val_test,
    discover_probe_examples,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    y_pred = (y_prob >= 0.5).astype(np.float32)
    return float(np.mean(y_pred == y_true))


def _f1_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    y_pred = (y_prob >= 0.5).astype(np.float32)
    tp = float(np.sum((y_pred == 1.0) & (y_true == 1.0)))
    fp = float(np.sum((y_pred == 1.0) & (y_true == 0.0)))
    fn = float(np.sum((y_pred == 0.0) & (y_true == 1.0)))
    if tp == 0.0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUROC using rank statistics without sklearn."""
    if y_true.size == 0:
        return float("nan")
    pos = y_true == 1.0
    neg = y_true == 0.0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    # Average ranks for ties.
    sorted_scores = y_score[order]
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            avg_rank = float(np.mean(np.arange(start + 1, end + 1, dtype=np.float64)))
            ranks[order[start:end]] = avg_rank
        start = end

    rank_sum_pos = float(np.sum(ranks[pos]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    while i < len(sorted_vals):
        j = i + 1
        while j < len(sorted_vals) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _spearman(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    if np.std(y_true) == 0.0 or np.std(y_score) == 0.0:
        return float("nan")
    return float(np.corrcoef(_rankdata(y_true), _rankdata(y_score))[0, 1])


def _mae(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_score)))


def _collect_predictions(
    model: LinearSchwartzProbe,
    examples,
    layer_name: str,
    pooling: int,
    device: torch.device,
):
    y_true_rows = []
    prob_rows = []
    binary_mask_rows = []
    continuous_mask_rows = []

    for ex in examples:
        try:
            hidden = _load_hidden_vector(ex, layer_name, pooling)
        except Exception:
            continue

        hidden_tensor = torch.tensor(hidden, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model(hidden_tensor).squeeze(0).cpu().numpy()
        probs = _sigmoid(logits)

        y_true_rows.append(ex.target)
        prob_rows.append(probs)
        binary_mask_rows.append(ex.binary_mask)
        continuous_mask_rows.append(ex.continuous_mask)

    if not y_true_rows:
        return None

    return (
        np.stack(y_true_rows),
        np.stack(prob_rows),
        np.stack(binary_mask_rows),
        np.stack(continuous_mask_rows),
    )


def _summarize_binary(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    if y_true.size == 0:
        return {"auroc": float("nan"), "f1": float("nan"), "accuracy": float("nan")}
    return {
        "auroc": _auroc(y_true, y_prob),
        "f1": _f1_score(y_true, y_prob),
        "accuracy": _accuracy(y_true, y_prob),
    }


def _summarize_continuous(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    if y_true.size == 0:
        return {"mae": float("nan"), "spearman": float("nan")}
    return {
        "mae": _mae(y_true, y_prob),
        "spearman": _spearman(y_true, y_prob),
    }


def _summarize_per_value_binary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    binary_mask: np.ndarray,
    value_names: List[str],
) -> Dict[str, Optional[Dict[str, float]]]:
    per_value: Dict[str, Optional[Dict[str, float]]] = {}
    for idx, value_name in enumerate(value_names):
        valid = binary_mask[:, idx] > 0.5
        value_true = y_true[valid, idx]
        value_prob = y_prob[valid, idx]
        if value_true.size == 0:
            per_value[value_name] = None
            continue
        metrics = _summarize_binary(value_true, value_prob)
        metrics["num_points"] = float(value_true.size)
        per_value[value_name] = metrics
    return per_value


def _summarize_per_value_continuous(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    continuous_mask: np.ndarray,
    value_names: List[str],
) -> Dict[str, Optional[Dict[str, float]]]:
    per_value: Dict[str, Optional[Dict[str, float]]] = {}
    for idx, value_name in enumerate(value_names):
        valid = continuous_mask[:, idx] > 0.5
        value_true = y_true[valid, idx]
        value_prob = y_prob[valid, idx]
        if value_true.size == 0:
            per_value[value_name] = None
            continue
        metrics = _summarize_continuous(value_true, value_prob)
        metrics["num_points"] = float(value_true.size)
        per_value[value_name] = metrics
    return per_value


def _print_metrics(title: str, metrics: Dict[str, float]):
    print(title)
    for key, value in metrics.items():
        if math.isnan(value):
            print(f"  {key:<10}: nan")
        else:
            print(f"  {key:<10}: {value:.4f}")


def _print_per_value_metrics(title: str, per_value_metrics: Dict[str, Optional[Dict[str, float]]]):
    print(title)
    for value_name, metrics in per_value_metrics.items():
        if not metrics:
            continue
        display_bits = []
        for metric_name, metric_value in metrics.items():
            if metric_name == "num_points":
                display_bits.append(f"{metric_name}={int(metric_value)}")
                continue
            if math.isnan(metric_value):
                display_bits.append(f"{metric_name}=nan")
            else:
                display_bits.append(f"{metric_name}={metric_value:.4f}")
        print(f"  {value_name:<14}: " + ", ".join(display_bits))


def _metric_or_nan(metrics: Dict[str, float], key: str) -> float:
    value = metrics.get(key, float("nan"))
    return float(value) if value is not None else float("nan")


def _sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return None
        return obj
    return obj


def _write_eval_outputs(rows: List[Dict], output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "probe_eval_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "checkpoint_path",
                "model_name",
                "layer_name",
                "pooling",
                "split",
                "split_source",
                "num_total_examples",
                "overall_binary_auroc",
                "overall_binary_f1",
                "overall_binary_accuracy",
                "overall_continuous_mae",
                "overall_continuous_spearman",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "num_runs": len(rows),
        "runs": rows,
    }
    json_path = output_dir / "probe_eval_results.json"
    json_path.write_text(json.dumps(_sanitize_for_json(summary), indent=2), encoding="utf-8")

    return csv_path, json_path


def _write_per_value_csvs(results: List[Dict], output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "binary": ("overall_binary_per_value", ["auroc", "f1", "accuracy", "num_points"]),
        "continuous": ("overall_continuous_per_value", ["mae", "spearman", "num_points"]),
    }
    paths: List[Path] = []
    for result in results:
        suffix = "" if len(results) == 1 else f"_{result['run_name']}"
        for kind, (key, metric_fields) in specs.items():
            path = output_dir / f"probe_eval_per_value_{kind}{suffix}.csv"
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["value"] + metric_fields)
                writer.writeheader()
                for value_name, metrics in (result.get(key) or {}).items():
                    if metrics:
                        writer.writerow({"value": value_name, **{m: metrics.get(m) for m in metric_fields}})
            paths.append(path)
    return paths


def _resolve_dataset_roots(
    *,
    valuenet_activations: Optional[str],
    valueeval_activations: Optional[str],
    fulcra_activations: Optional[str],
    activations_cfg: Optional[Dict],
) -> Dict[str, Path]:
    dataset_roots: Dict[str, Path] = {}

    if valuenet_activations:
        dataset_roots["valuenet"] = Path(valuenet_activations)
    if valueeval_activations:
        dataset_roots["valueeval"] = Path(valueeval_activations)
    if fulcra_activations:
        dataset_roots["fulcra"] = Path(fulcra_activations)

    if activations_cfg:
        if "valuenet" in activations_cfg:
            dataset_roots["valuenet"] = Path(activations_cfg["valuenet"])
        if "valueeval" in activations_cfg:
            dataset_roots["valueeval"] = Path(activations_cfg["valueeval"])
        if "fulcra" in activations_cfg:
            dataset_roots["fulcra"] = Path(activations_cfg["fulcra"])

    if not dataset_roots:
        dataset_roots = {
            "valuenet": Path("activations/valuenet"),
            "valueeval": Path("activations/valueeval"),
            "fulcra": Path("activations/fulcra"),
        }

    return dataset_roots


def _evaluate_one(
    *,
    checkpoint_path: Path,
    split: str,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    device_override: Optional[str],
    layer_name_override: Optional[str],
    pooling_override: Optional[int],
    dataset_roots: Dict[str, Path],
    run_name: str,
) -> Tuple[int, Dict]:
    if not checkpoint_path.exists():
        print(f"Error: checkpoint not found: {checkpoint_path}")
        return 1, {}

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    layer_name = layer_name_override or checkpoint["layer_name"]
    pooling = int(pooling_override if pooling_override is not None else checkpoint["pooling"])

    examples_by_dataset = {}
    for dataset_name, root in dataset_roots.items():
        examples_by_dataset[dataset_name] = discover_probe_examples(dataset_name, root)

    non_empty = {name: items for name, items in examples_by_dataset.items() if items}
    if not non_empty:
        print("Error: no activation examples found in the provided dataset roots")
        return 1, {}

    split_manifest = checkpoint.get("split_manifest")
    print(split_manifest['valueeval']['val'])
    if split_manifest:
        eval_split: Dict[str, List] = {}
        selected_counts = {}
        manifest_errors: List[str] = []
        for dataset_name, examples in examples_by_dataset.items():
            root = dataset_roots[dataset_name]
            by_rel = {
                _safe_relpath(ex.activation_file, root): ex
                for ex in examples
            }
            dataset_manifest = split_manifest.get(dataset_name)
            if dataset_manifest is None:
                manifest_errors.append(
                    f"- Dataset '{dataset_name}' is not present in checkpoint split manifest"
                )
                eval_split[dataset_name] = []
                selected_counts[dataset_name] = 0
                continue

            if split not in dataset_manifest:
                manifest_errors.append(
                    f"- Split '{split}' missing for dataset '{dataset_name}' in checkpoint manifest"
                )
                eval_split[dataset_name] = []
                selected_counts[dataset_name] = 0
                continue

            manifest_entries = dataset_manifest.get(split, [])
            selected_examples = [by_rel[rel] for rel in manifest_entries if rel in by_rel]
            eval_split[dataset_name] = selected_examples
            selected_counts[dataset_name] = len(selected_examples)

            if manifest_entries and not selected_examples:
                manifest_errors.append(
                    f"- Dataset '{dataset_name}' split '{split}' has {len(manifest_entries)} manifest entries, but 0 matched under root '{root}'"
                )

        if manifest_errors:
            print("Error: checkpoint split manifest could not be resolved with the provided activation roots")
            for msg in manifest_errors:
                print(msg)
            print("Hint: pass activation roots that match the checkpoint's processed data location.")
            return 1, {}

        if sum(selected_counts.values()) == 0:
            print(
                f"Error: checkpoint manifest split '{split}' resolved to zero total examples for the provided roots"
            )
            return 1, {}
        split_source = "checkpoint_manifest"
        split_seed = checkpoint.get("split_seed", "n/a")
        split_val_ratio = checkpoint.get("val_ratio", "n/a")
        split_test_ratio = checkpoint.get("test_ratio", "n/a")
    else:
        # Backward compatibility for old checkpoints that don't store split manifests.
        train_split, val_split, test_split = _split_train_val_test(
            examples_by_dataset,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        split_map = {"train": train_split, "val": val_split, "test": test_split}
        eval_split = split_map[split]
        selected_counts = {name: len(items) for name, items in eval_split.items()}
        split_source = "legacy_runtime_fallback"
        split_seed = seed
        split_val_ratio = val_ratio
        split_test_ratio = test_ratio

    device = torch.device(device_override or ("cuda" if torch.cuda.is_available() else "cpu"))
    schwartz_values = list(checkpoint.get("schwartz_values", []))
    if schwartz_values:
        num_values = len(schwartz_values)
    else:
        num_values = len(SCHWARTZ_10_VALUES)
        schwartz_values = list(SCHWARTZ_10_VALUES)
    model = LinearSchwartzProbe(hidden_size=int(checkpoint["hidden_size"]), num_values=num_values).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    print("=" * 80)
    print("Schwartz Probe Evaluation")
    print("=" * 80)
    print(f"Run        : {run_name}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Layer      : {layer_name}")
    print(f"Pooling    : {pooling}")
    print(f"Device     : {device}")
    print(f"Split      : {split}")
    if split_manifest:
        print("Split source: checkpoint manifest")
        print(f"Split seed  : {checkpoint.get('split_seed', 'n/a')}")
        print(f"Val ratio   : {checkpoint.get('val_ratio', 'n/a')}")
        print(f"Test ratio  : {checkpoint.get('test_ratio', 'n/a')}")
    else:
        print("Split source: runtime fallback (legacy checkpoint)")
        print(f"Split seed  : {seed}")
        print(f"Val ratio   : {val_ratio}")
        print(f"Test ratio  : {test_ratio}")
    print("=" * 80)
    print("Dataset split sizes:")
    for name, count in selected_counts.items():
        print(f"  {name:<10}: {count}")
    print("=" * 80)

    overall_true_rows = []
    overall_prob_rows = []
    overall_binary_mask_rows = []
    overall_cont_mask_rows = []

    dataset_metrics: Dict[str, Dict] = {}

    for dataset_name, examples in eval_split.items():
        if not examples:
            continue
        packed = _collect_predictions(model, examples, layer_name=layer_name, pooling=pooling, device=device)
        if packed is None:
            continue

        y_true, y_prob, binary_mask, continuous_mask = packed

        b_true = y_true[binary_mask > 0.5]
        b_prob = y_prob[binary_mask > 0.5]
        c_true = y_true[continuous_mask > 0.5]
        c_prob = y_prob[continuous_mask > 0.5]

        bin_metrics = _summarize_binary(b_true, b_prob)
        cont_metrics = _summarize_continuous(c_true, c_prob)
        per_value_binary = _summarize_per_value_binary(y_true, y_prob, binary_mask, schwartz_values)
        per_value_continuous = _summarize_per_value_continuous(
            y_true,
            y_prob,
            continuous_mask,
            schwartz_values,
        )

        dataset_metrics[dataset_name] = {
            "num_examples": int(len(examples)),
            "binary": bin_metrics if b_true.size > 0 else None,
            "continuous": cont_metrics if c_true.size > 0 else None,
            "binary_per_value": per_value_binary,
            "continuous_per_value": per_value_continuous,
        }

        print(f"\n[{dataset_name}]")
        if b_true.size > 0:
            _print_metrics("Binary", bin_metrics)
            _print_per_value_metrics("Binary per-value", per_value_binary)
        if c_true.size > 0:
            _print_metrics("Continuous", cont_metrics)
            _print_per_value_metrics("Continuous per-value", per_value_continuous)

        overall_true_rows.append(y_true)
        overall_prob_rows.append(y_prob)
        overall_binary_mask_rows.append(binary_mask)
        overall_cont_mask_rows.append(continuous_mask)

    overall_binary_metrics: Optional[Dict[str, float]] = None
    overall_cont_metrics: Optional[Dict[str, float]] = None
    overall_binary_per_value: Dict[str, Optional[Dict[str, float]]] = {
        value_name: None for value_name in schwartz_values
    }
    overall_cont_per_value: Dict[str, Optional[Dict[str, float]]] = {
        value_name: None for value_name in schwartz_values
    }

    if overall_true_rows:
        y_true_all = np.concatenate(overall_true_rows, axis=0)
        y_prob_all = np.concatenate(overall_prob_rows, axis=0)
        binary_mask_all = np.concatenate(overall_binary_mask_rows, axis=0)
        b_true = y_true_all[binary_mask_all > 0.5]
        b_prob = y_prob_all[binary_mask_all > 0.5]

        if b_true.size > 0:
            overall_binary_metrics = _summarize_binary(b_true, b_prob)
            overall_binary_per_value = _summarize_per_value_binary(
                y_true_all,
                y_prob_all,
                binary_mask_all,
                schwartz_values,
            )
            print("\n[Overall binary]")
            _print_metrics("Binary", overall_binary_metrics)
            _print_per_value_metrics("Binary per-value", overall_binary_per_value)

    if overall_true_rows:
        y_true_all = np.concatenate(overall_true_rows, axis=0)
        y_prob_all = np.concatenate(overall_prob_rows, axis=0)
        cont_mask_all = np.concatenate(overall_cont_mask_rows, axis=0)
        c_true = y_true_all[cont_mask_all > 0.5]
        c_prob = y_prob_all[cont_mask_all > 0.5]

        if c_true.size > 0:
            overall_cont_metrics = _summarize_continuous(c_true, c_prob)
            overall_cont_per_value = _summarize_per_value_continuous(
                y_true_all,
                y_prob_all,
                cont_mask_all,
                schwartz_values,
            )
            print("\n[Overall continuous]")
            _print_metrics("Continuous", overall_cont_metrics)
            _print_per_value_metrics("Continuous per-value", overall_cont_per_value)

    print("\nDone.")

    result = {
        "run_name": run_name,
        "checkpoint_path": str(checkpoint_path),
        "model_name": str(checkpoint.get("model_name", "unknown")),
        "layer_name": layer_name,
        "pooling": int(pooling),
        "split": split,
        "split_source": split_source,
        "split_seed": split_seed,
        "val_ratio": split_val_ratio,
        "test_ratio": split_test_ratio,
        "device": str(device),
        "dataset_counts": {k: int(v) for k, v in selected_counts.items()},
        "num_total_examples": int(sum(selected_counts.values())),
        "dataset_metrics": dataset_metrics,
        "overall_binary": overall_binary_metrics,
        "overall_continuous": overall_cont_metrics,
        "overall_binary_per_value": overall_binary_per_value,
        "overall_continuous_per_value": overall_cont_per_value,
    }
    return 0, result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained Schwartz probe on held-out activations")
    parser.add_argument("--checkpoint", default=None, help="Path to a saved probe checkpoint (.pt)")
    parser.add_argument("--config", default=None, help="Path to JSON config for multi-checkpoint evaluation")
    parser.add_argument("--output-dir", default=None, help="Optional directory to write parsable CSV/JSON results")
    parser.add_argument("--run-name", default="single_eval", help="Run label for single-checkpoint evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the held-out split")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of each dataset reserved for validation")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Fraction of each dataset reserved for test")
    parser.add_argument(
        "--split",
        choices=["test", "val", "train"],
        default="test",
        help="Which split to evaluate from the checkpoint manifest (default: test)",
    )
    parser.add_argument("--device", default=None, help="Optional device override (cuda, cpu)")
    parser.add_argument("--layer-name", default=None, help="Override checkpoint layer name for evaluation")
    parser.add_argument("--pooling", type=int, default=None, help="Override checkpoint pooling for evaluation")

    parser.add_argument("--valuenet-activations", default=None, help="Path to ValueNet activation root")
    parser.add_argument("--valueeval-activations", default=None, help="Path to ValueEval activation root")
    parser.add_argument("--fulcra-activations", default=None, help="Path to FULCRA activation root")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows: List[Dict] = []
    results: List[Dict] = []

    if args.config:
        config_path = Path(args.config)
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        shared = cfg.get("shared", {})
        evaluations = cfg.get("evaluations", [])
        if not evaluations:
            print("Error: config must contain a non-empty 'evaluations' list")
            return 1

        shared_output_dir = shared.get("output_dir")
        output_dir = Path(args.output_dir or shared_output_dir or "artifacts/probe_eval")

        for item in evaluations:
            run_name = str(item.get("name", Path(str(item.get("checkpoint", "run"))).stem))
            checkpoint_path = Path(str(item.get("checkpoint", "")))
            if not str(checkpoint_path):
                print(f"Error: evaluation run '{run_name}' has no checkpoint")
                return 1

            split = str(item.get("split", shared.get("split", args.split)))
            seed = int(item.get("seed", shared.get("seed", args.seed)))
            val_ratio = float(item.get("val_ratio", shared.get("val_ratio", args.val_ratio)))
            test_ratio = float(item.get("test_ratio", shared.get("test_ratio", args.test_ratio)))
            device = item.get("device", shared.get("device", args.device))
            layer_name = item.get("layer_name", shared.get("layer_name", args.layer_name))
            pooling = item.get("pooling", shared.get("pooling", args.pooling))
            activations_cfg = item.get("activations", shared.get("activations"))

            dataset_roots = _resolve_dataset_roots(
                valuenet_activations=args.valuenet_activations,
                valueeval_activations=args.valueeval_activations,
                fulcra_activations=args.fulcra_activations,
                activations_cfg=activations_cfg,
            )

            code, result = _evaluate_one(
                checkpoint_path=checkpoint_path,
                split=split,
                seed=seed,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                device_override=device,
                layer_name_override=layer_name,
                pooling_override=pooling,
                dataset_roots=dataset_roots,
                run_name=run_name,
            )
            if code != 0:
                return code

            results.append(result)
            rows.append(
                {
                    "run_name": run_name,
                    "checkpoint_path": result["checkpoint_path"],
                    "model_name": result.get("model_name", "unknown"),
                    "layer_name": result["layer_name"],
                    "pooling": result["pooling"],
                    "split": result["split"],
                    "split_source": result["split_source"],
                    "num_total_examples": result["num_total_examples"],
                    "overall_binary_auroc": _metric_or_nan(result.get("overall_binary") or {}, "auroc"),
                    "overall_binary_f1": _metric_or_nan(result.get("overall_binary") or {}, "f1"),
                    "overall_binary_accuracy": _metric_or_nan(result.get("overall_binary") or {}, "accuracy"),
                    "overall_continuous_mae": _metric_or_nan(result.get("overall_continuous") or {}, "mae"),
                    "overall_continuous_spearman": _metric_or_nan(result.get("overall_continuous") or {}, "spearman"),
                }
            )

        csv_path, json_path = _write_eval_outputs(rows, output_dir)
        per_value_paths = _write_per_value_csvs(results, output_dir)
        print("\nEvaluation sweep complete")
        print(f"Results CSV : {csv_path}")
        print(f"Results JSON: {json_path}")
        for p in per_value_paths:
            print(f"Per-value CSV: {p}")
        return 0

    if not args.checkpoint:
        print("Error: --checkpoint is required when --config is not provided")
        return 1

    dataset_roots = _resolve_dataset_roots(
        valuenet_activations=args.valuenet_activations,
        valueeval_activations=args.valueeval_activations,
        fulcra_activations=args.fulcra_activations,
        activations_cfg=None,
    )

    code, result = _evaluate_one(
        checkpoint_path=Path(args.checkpoint),
        split=args.split,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        device_override=args.device,
        layer_name_override=args.layer_name,
        pooling_override=args.pooling,
        dataset_roots=dataset_roots,
        run_name=args.run_name,
    )
    if code != 0:
        return code

    if args.output_dir:
        rows.append(
            {
                "run_name": args.run_name,
                "checkpoint_path": result["checkpoint_path"],
                "model_name": result.get("model_name", "unknown"),
                "layer_name": result["layer_name"],
                "pooling": result["pooling"],
                "split": result["split"],
                "split_source": result["split_source"],
                "num_total_examples": result["num_total_examples"],
                "overall_binary_auroc": _metric_or_nan(result.get("overall_binary") or {}, "auroc"),
                "overall_binary_f1": _metric_or_nan(result.get("overall_binary") or {}, "f1"),
                "overall_binary_accuracy": _metric_or_nan(result.get("overall_binary") or {}, "accuracy"),
                "overall_continuous_mae": _metric_or_nan(result.get("overall_continuous") or {}, "mae"),
                "overall_continuous_spearman": _metric_or_nan(result.get("overall_continuous") or {}, "spearman"),
            }
        )
        csv_path, json_path = _write_eval_outputs(rows, Path(args.output_dir))
        per_value_paths = _write_per_value_csvs([result], Path(args.output_dir))
        print(f"Saved parsable results to: {csv_path} and {json_path}")
        for p in per_value_paths:
            print(f"Saved per-value results to: {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

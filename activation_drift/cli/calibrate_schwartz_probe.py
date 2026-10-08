#!/usr/bin/env python3
"""Compute per-value logit calibration stats (mu, sigma) for a trained Schwartz probe.

This script freezes a probe checkpoint and computes, for each Schwartz value label,
- mean logit (mu)
- std logit (sigma)

on a chosen split (recommended: train) so downstream analysis can use
standardized pair scores:
    z_k = (logit_k - mu_k) / sigma_k
    A_t_std(value_a, value_b) = z_a - z_b
"""

from __future__ import annotations

import argparse
import json
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


def _resolve_dataset_roots(
    *,
    valuenet_activations: Optional[str],
    valueeval_activations: Optional[str],
    fulcra_activations: Optional[str],
) -> Dict[str, Path]:
    dataset_roots: Dict[str, Path] = {}
    if valuenet_activations:
        dataset_roots["valuenet"] = Path(valuenet_activations)
    if valueeval_activations:
        dataset_roots["valueeval"] = Path(valueeval_activations)
    if fulcra_activations:
        dataset_roots["fulcra"] = Path(fulcra_activations)

    if not dataset_roots:
        dataset_roots = {
            "valuenet": Path("activations/valuenet"),
            "valueeval": Path("activations/valueeval"),
            "fulcra": Path("activations/fulcra"),
        }
    return dataset_roots


def _select_split_examples(
    *,
    checkpoint: dict,
    checkpoint_path: Path,
    split: str,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    dataset_roots: Dict[str, Path],
) -> Tuple[Dict[str, List], str, Dict[str, int]]:
    examples_by_dataset = {
        name: discover_probe_examples(name, root) for name, root in dataset_roots.items()
    }
    non_empty = {name: items for name, items in examples_by_dataset.items() if items}
    if not non_empty:
        raise ValueError("No activation examples found in the provided dataset roots")

    split_manifest = checkpoint.get("split_manifest")
    if split_manifest:
        eval_split: Dict[str, List] = {}
        selected_counts: Dict[str, int] = {}
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
                    f"- Dataset '{dataset_name}' split '{split}' has {len(manifest_entries)} entries, "
                    f"but 0 matched under root '{root}'"
                )

        if manifest_errors:
            raise ValueError(
                "Checkpoint split manifest could not be resolved with the provided activation roots:\n"
                + "\n".join(manifest_errors)
            )
        if sum(selected_counts.values()) == 0:
            raise ValueError(
                f"Checkpoint manifest split '{split}' resolved to zero examples"
            )
        return eval_split, "checkpoint_manifest", selected_counts

    # Legacy fallback for checkpoints without a stored manifest.
    train_split, val_split, test_split = _split_train_val_test(
        examples_by_dataset,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    split_map = {"train": train_split, "val": val_split, "test": test_split}
    selected = split_map[split]
    counts = {name: len(items) for name, items in selected.items()}
    if sum(counts.values()) == 0:
        raise ValueError(
            f"Legacy runtime split '{split}' resolved to zero examples for checkpoint {checkpoint_path}"
        )
    return selected, "legacy_runtime_fallback", counts


def _collect_logits(
    *,
    model: LinearSchwartzProbe,
    examples_by_dataset: Dict[str, List],
    layer_name: str,
    pooling: int,
    device: torch.device,
) -> Tuple[np.ndarray, List[str]]:
    logits_rows: List[np.ndarray] = []
    dataset_rows: List[str] = []

    for dataset_name, examples in examples_by_dataset.items():
        for ex in examples:
            try:
                hidden = _load_hidden_vector(ex, layer_name, pooling)
            except Exception:
                continue

            hidden_tensor = torch.tensor(hidden, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = model(hidden_tensor).squeeze(0).cpu().numpy()
            logits_rows.append(logits)
            dataset_rows.append(dataset_name)

    if not logits_rows:
        raise ValueError("No logits were collected (all examples failed to load layer/pooling)")

    return np.stack(logits_rows), dataset_rows


def _value_stats(
    logits: np.ndarray,
    value_names: List[str],
    std_floor: float,
) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(value_names):
        col = logits[:, i]
        mu = float(np.mean(col))
        sigma_raw = float(np.std(col))
        sigma_used = sigma_raw if sigma_raw >= std_floor else float(std_floor)
        stats[name] = {
            "mu": mu,
            "sigma": sigma_used,
            "sigma_raw": sigma_raw,
            "sigma_floored": bool(sigma_raw < std_floor),
        }
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute per-value probe logit calibration stats (mu, sigma) for a checkpoint"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a trained probe checkpoint (.pt)")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Split used to estimate calibration stats (recommended: train)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Legacy split seed fallback")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Legacy fallback val ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Legacy fallback test ratio")
    parser.add_argument("--device", default=None, help="Optional device override (cuda, cpu)")
    parser.add_argument("--layer-name", default=None, help="Optional override layer name")
    parser.add_argument("--pooling", type=int, default=None, help="Optional override pooling")
    parser.add_argument("--std-floor", type=float, default=1e-6, help="Minimum sigma to avoid divide-by-zero")
    parser.add_argument("--output", default=None, help="Output JSON path (default: alongside checkpoint)")

    parser.add_argument("--valuenet-activations", default=None, help="Path to ValueNet activation root")
    parser.add_argument("--valueeval-activations", default=None, help="Path to ValueEval activation root")
    parser.add_argument("--fulcra-activations", default=None, help="Path to FULCRA activation root")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    layer_name = args.layer_name or checkpoint["layer_name"]
    pooling = int(args.pooling if args.pooling is not None else checkpoint["pooling"])
    schwartz_values = list(checkpoint.get("schwartz_values", []))
    if not schwartz_values:
        schwartz_values = list(SCHWARTZ_10_VALUES)
    num_values = len(schwartz_values)

    dataset_roots = _resolve_dataset_roots(
        valuenet_activations=args.valuenet_activations,
        valueeval_activations=args.valueeval_activations,
        fulcra_activations=args.fulcra_activations,
    )

    selected_examples, split_source, selected_counts = _select_split_examples(
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        split=args.split,
        seed=int(args.seed),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        dataset_roots=dataset_roots,
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = LinearSchwartzProbe(hidden_size=int(checkpoint["hidden_size"]), num_values=num_values).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    logits, dataset_rows = _collect_logits(
        model=model,
        examples_by_dataset=selected_examples,
        layer_name=layer_name,
        pooling=pooling,
        device=device,
    )

    per_value = _value_stats(logits, schwartz_values, std_floor=float(args.std_floor))

    per_dataset: Dict[str, Dict[str, float]] = {}
    ds_arr = np.array(dataset_rows)
    for ds_name in sorted(set(dataset_rows)):
        mask = ds_arr == ds_name
        ds_logits = logits[mask]
        per_dataset[ds_name] = {
            "num_examples": int(ds_logits.shape[0]),
            "logit_mean_global": float(np.mean(ds_logits)),
            "logit_std_global": float(np.std(ds_logits)),
        }

    output_path = (
        Path(args.output)
        if args.output
        else checkpoint_path.with_name(f"{checkpoint_path.stem}_calibration_{args.split}.json")
    )

    payload = {
        "checkpoint_path": str(checkpoint_path),
        "model_name": str(checkpoint.get("model_name", "unknown")),
        "layer_name": str(layer_name),
        "pooling": int(pooling),
        "split": str(args.split),
        "split_source": split_source,
        "split_seed": int(checkpoint.get("split_seed", args.seed)),
        "val_ratio": float(checkpoint.get("val_ratio", args.val_ratio)),
        "test_ratio": float(checkpoint.get("test_ratio", args.test_ratio)),
        "std_floor": float(args.std_floor),
        "device": str(device),
        "dataset_roots": {k: str(v) for k, v in dataset_roots.items()},
        "dataset_selected_counts": {k: int(v) for k, v in selected_counts.items()},
        "num_examples_used": int(logits.shape[0]),
        "schwartz_values": schwartz_values,
        "per_value": per_value,
        "per_dataset": per_dataset,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=" * 80)
    print("Probe Logit Calibration")
    print("=" * 80)
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Layer      : {layer_name}")
    print(f"Pooling    : {pooling}")
    print(f"Split      : {args.split} ({split_source})")
    print(f"Examples   : {logits.shape[0]}")
    print(f"Output     : {output_path}")
    print("=" * 80)
    print("Per-value mu/sigma:")
    for value_name in schwartz_values:
        s = per_value[value_name]
        marker = " *" if s["sigma_floored"] else ""
        print(
            f"  {value_name:<14} mu={s['mu']:+.6f} sigma={s['sigma']:.6f} "
            f"(raw={s['sigma_raw']:.6f}){marker}"
        )
    if any(v["sigma_floored"] for v in per_value.values()):
        print("* sigma was floored to std_floor for at least one value")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

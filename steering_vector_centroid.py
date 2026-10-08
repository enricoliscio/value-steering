#!/usr/bin/env python3
"""Compute expectation-weighted Schwartz value centroids from saved activations.

For each of the 10 Schwartz values v, computes:

    mu_v = sum_i w_i(v) * h_i / sum_i w_i(v)

where w_i(v) is the example's score for value v (continuous target in [0,1],
or binary mask * target if --labeled-only is set). Pairwise steering vectors
for any (a, b) are just mu_a - mu_b, derived post-hoc from the saved centroids
via pairwise_vector_from_centroids() -- no re-scanning activations needed.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from activation_drift.schwartz_probe import (
    ProbeExample,
    _load_hidden_vector,
    _safe_relpath,
    discover_probe_examples,
)

VALUE_ORDER = [
    "SELF-DIRECTION", "STIMULATION", "HEDONISM", "ACHIEVEMENT", "POWER",
    "SECURITY", "TRADITION", "CONFORMITY", "BENEVOLENCE", "UNIVERSALISM",
]


def _layer_tag(layer_name: str) -> str:
    m = re.search(r"(\d+)$", str(layer_name).strip())
    return m.group(1) if m else re.sub(r"[^A-Za-z0-9_.-]+", "_", str(layer_name).strip())


def _example_weights(ex: ProbeExample, labeled_only: bool) -> np.ndarray:
    """Return a length-10 weight vector for this example.

    labeled_only=True  -> w[v] = target[v] if (binary_mask[v] or continuous_mask[v]) else 0
    labeled_only=False -> w[v] = target[v] for all v (missing treated as 0 = "not expressed")
    """
    target = np.asarray(ex.target, dtype=np.float64)
    if not labeled_only:
        return target
    labeled = (np.asarray(ex.binary_mask) > 0.5) | (np.asarray(ex.continuous_mask) > 0.5)
    return target * labeled.astype(np.float64)


def _apply_manifest_split(examples_by_dataset, dataset_roots, split_manifest, split):
    selected = {}
    for ds_name, examples in examples_by_dataset.items():
        ds_manifest = split_manifest.get(ds_name, {}) if isinstance(split_manifest, dict) else {}
        wanted = ds_manifest.get(split, []) if isinstance(ds_manifest, dict) else []
        by_rel = {_safe_relpath(ex.activation_file, dataset_roots[ds_name]): ex for ex in examples}
        selected[ds_name] = [by_rel[r] for r in wanted if r in by_rel]
    return selected


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute expectation-weighted Schwartz value centroids")
    p.add_argument("--layer-name", default=None)
    p.add_argument("--model-name", default="llama8b", help="Tagged into output filename + metadata")
    p.add_argument("--pooling", type=int, default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--split", choices=["all", "train", "val", "test"], default="train")
    p.add_argument("--valuenet-activations", default="activations/valuenet")
    p.add_argument("--valueeval-activations", default="activations/valueeval")
    p.add_argument("--fulcra-activations", default="activations/fulcra")
    p.add_argument("--datasets", nargs="+", choices=["valuenet", "valueeval", "fulcra"],
                    default=["valuenet", "valueeval", "fulcra"])
    p.add_argument(
        "--labeled-only", action="store_true",
        help="Zero out weight for a value on examples where it has neither binary nor "
             "continuous label (instead of treating missing as 0 = 'not expressed')",
    )
    p.add_argument("--normalize", action="store_true", help="L2-normalize each stored centroid")
    p.add_argument("--output-dir", default="steering_vectors_centroid")
    p.add_argument("--progress-every", type=int, default=1000,
                    help="Print a progress line every N examples seen (0 to disable)")
    return p


def main() -> int:
    args = build_parser().parse_args()

    checkpoint = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Fail instead of silently using all examples when there's no manifest to split by.
    if args.split != "all" and not (checkpoint and checkpoint.get("split_manifest")):
        raise ValueError(
            f"--split {args.split} requires --checkpoint with a saved split_manifest; "
            "got none, so there is no train/val/test membership to apply. "
            "Pass --split all, or a checkpoint that has one."
        )

    layer_name = args.layer_name or (checkpoint.get("layer_name") if checkpoint else None)
    pooling = int(args.pooling if args.pooling is not None else (checkpoint.get("pooling") if checkpoint else 1))
    if not layer_name:
        raise ValueError("Provide --layer-name or pass --checkpoint with layer metadata")

    dataset_roots_all = {
        "valuenet": Path(args.valuenet_activations),
        "valueeval": Path(args.valueeval_activations),
        "fulcra": Path(args.fulcra_activations),
    }
    dataset_roots = {k: dataset_roots_all[k] for k in args.datasets}

    examples_by_dataset: Dict[str, List[ProbeExample]] = {
        ds: discover_probe_examples(ds, root) for ds, root in dataset_roots.items()
    }

    if args.split != "all":
        examples_by_dataset = _apply_manifest_split(
            examples_by_dataset, dataset_roots, checkpoint.get("split_manifest", {}), args.split
        )

    n_values = len(VALUE_ORDER)
    weighted_sum = None       # (10, D)
    weight_total = np.zeros(n_values, dtype=np.float64)
    n_seen, n_used, n_skipped_load = 0, 0, 0

    for ds_name, ds_examples in examples_by_dataset.items():
        for ex in ds_examples:
            n_seen += 1
            if args.progress_every and n_seen % args.progress_every == 0:
                print(f"...seen {n_seen} examples ({n_used} used so far)")

            has_any_label = (
                np.any(np.asarray(ex.binary_mask) > 0.5)
                or np.any(np.asarray(ex.continuous_mask) > 0.5)
            )
            if not has_any_label:
                continue  # no label info at all; skip entirely

            w = _example_weights(ex, labeled_only=args.labeled_only)  # (10,)
            if not np.any(w > 0):
                continue  # labeled_only=True and none of its labeled values scored > 0

            try:
                hidden = _load_hidden_vector(ex, layer_name=layer_name, pooling=pooling)
            except Exception:
                n_skipped_load += 1
                continue

            hidden = hidden.astype(np.float64)
            if weighted_sum is None:
                weighted_sum = np.zeros((n_values, hidden.shape[0]), dtype=np.float64)

            weighted_sum += np.outer(w, hidden)
            weight_total += w
            n_used += 1

    if weighted_sum is None or n_used == 0:
        raise ValueError(f"No usable examples found (seen={n_seen}, used={n_used}).")

    centroids: Dict[str, np.ndarray] = {}
    for i, value in enumerate(VALUE_ORDER):
        if weight_total[i] <= 0:
            centroids[value] = None
            continue
        mu_v = (weighted_sum[i] / weight_total[i]).astype(np.float32)
        if args.normalize:
            norm = float(np.linalg.norm(mu_v))
            if norm > 0:
                mu_v = (mu_v / norm).astype(np.float32)
        centroids[value] = mu_v

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_slug = _layer_tag(layer_name)
    base_name = f"schwartz_centroids_{args.model_name}_l{layer_slug}_p{pooling}"

    npz_payload = {f"centroid_{v}": c for v, c in centroids.items() if c is not None}
    npz_path = output_dir / f"{base_name}.npz"
    np.savez(npz_path, **npz_payload)

    meta = {
        "created_at": datetime.now().isoformat(),
        "model_name": args.model_name,
        "layer_name": layer_name,
        "pooling": pooling,
        "split": args.split,
        "checkpoint": args.checkpoint,
        "datasets": list(dataset_roots.keys()),
        "labeled_only": args.labeled_only,
        "normalized": args.normalize,
        "n_seen": n_seen,
        "n_used": n_used,
        "n_skipped_load": n_skipped_load,
        "weight_total": {v: float(w) for v, w in zip(VALUE_ORDER, weight_total)},
        "values_with_data": [v for v, c in centroids.items() if c is not None],
        "output_npz": str(npz_path),
    }
    meta_path = output_dir / f"{base_name}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("=" * 80)
    print("Computed expectation-weighted Schwartz centroids")
    print(f"model         : {args.model_name}")
    print(f"layer/pooling : {layer_name} / {pooling}")
    print(f"seen/used     : {n_seen} / {n_used}  (skipped_load={n_skipped_load})")
    for v in VALUE_ORDER:
        wt = weight_total[VALUE_ORDER.index(v)]
        flag = "(no data)" if wt <= 0 else ""
        print(f"  {v:<14} weight_total={wt:.2f}  {flag}")
    print(f"saved: {npz_path}")
    print(f"saved: {meta_path}")
    print("=" * 80)
    return 0


def pairwise_vector_from_centroids(npz_path: str, value_a: str, value_b: str) -> np.ndarray:
    """Derive an a-vs-b steering vector from stored per-value centroids."""
    data = np.load(npz_path)
    key_a, key_b = f"centroid_{value_a}", f"centroid_{value_b}"
    if key_a not in data or key_b not in data:
        raise KeyError(f"Missing centroid for {value_a} or {value_b} in {npz_path}")
    return (data[key_a] - data[key_b]).astype(np.float32)


if __name__ == "__main__":
    raise SystemExit(main())
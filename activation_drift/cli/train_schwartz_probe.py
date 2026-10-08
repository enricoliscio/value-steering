#!/usr/bin/env python3
"""Train Schwartz probes (single layer or layer sweep) from saved activations.

Supports:
- single-layer training (--layer-name)
- layer sweep for one model (--layers ... --model-name ...)
- multi-model, multi-layer sweeps via JSON config (--config)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from activation_drift import train_linear_schwartz_probe


def _normalize_layer_spec(layer: str) -> str:
    text = str(layer).strip()
    if text.isdigit():
        return f"model.layers.{int(text)}"
    return text


def _layer_index(layer_name: str) -> str:
    match = re.search(r"(\d+)$", layer_name)
    return match.group(1) if match else layer_name.replace(".", "_")


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())


def _checkpoint_path(output: Path, model_name: str, layer_name: str, multi_run: bool) -> Path:
    if not multi_run:
        return output

    if output.suffix == ".pt":
        out_dir = output.parent
        stem = output.stem
    else:
        out_dir = output
        stem = "schwartz_probe"

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}_{_safe_name(model_name)}_l{_layer_index(layer_name)}.pt"


def _run_single(
        *,
        model_name: str,
        dataset_roots: Dict[str, Path],
        output_path: Path,
        layer_name: str,
        pooling: int,
        batch_size: int,
        epochs: int,
        val_ratio: float,
        test_ratio: float,
        lr: float,
        weight_decay: float,
        lambda_bin: float,
        lambda_cont: float,
        seed: int,
) -> Dict:
    summary = train_linear_schwartz_probe(
        dataset_roots=dataset_roots,
        output_path=output_path,
        layer_name=layer_name,
        pooling=pooling,
        batch_size=batch_size,
        epochs=epochs,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        lr=lr,
        weight_decay=weight_decay,
        lambda_bin=lambda_bin,
        lambda_cont=lambda_cont,
        seed=seed,
    )

    return {
        "model_name": model_name,
        "layer_name": layer_name,
        "pooling": pooling,
        "best_epoch": int(summary["best_epoch"]),
        "best_val_loss": float(summary["best_val_loss"]),
        "num_train_examples": int(summary["num_train_examples"]),
        "num_val_examples": int(summary["num_val_examples"]),
        "num_test_examples": int(summary["num_test_examples"]),
        "checkpoint_path": str(output_path),
    }


def _print_header(title: str):
    print("=" * 80)
    print(title)
    print("=" * 80)


def _write_sweep_summaries(rows: List[Dict], output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "probe_sweep_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_name",
                "layer_name",
                "pooling",
                "best_epoch",
                "best_val_loss",
                "num_train_examples",
                "num_val_examples",
                "num_test_examples",
                "checkpoint_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    by_model: Dict[str, Dict] = {}
    for row in rows:
        m = row["model_name"]
        if m not in by_model or float(row["best_val_loss"]) < float(by_model[m]["best_val_loss"]):
            by_model[m] = row

    summary = {
        "num_runs": len(rows),
        "best_by_model": by_model,
        "all_runs": rows,
    }
    json_path = output_dir / "probe_sweep_results.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return csv_path, json_path


def _train_one_model_layer_sweep(args) -> int:
    dataset_roots = {
        "valueeval": Path(args.valueeval_activations),
        "valuenet": Path(args.valuenet_activations),
        "fulcra": Path(args.fulcra_activations),
    }

    layers = [_normalize_layer_spec(x) for x in (args.layers or [args.layer_name])]
    multi_run = len(layers) > 1

    _print_header("Training Schwartz Probe (Single Model)")
    print(f"Model name           : {args.model_name}")
    print(f"ValueEval activations: {dataset_roots['valueeval']}")
    print(f"ValueNet activations : {dataset_roots['valuenet']}")
    print(f"FULCRA activations   : {dataset_roots['fulcra']}")
    print(f"Layers               : {', '.join(layers)}")
    print(f"Pooling              : {args.pooling}")
    print(f"Batch size           : {args.batch_size}")
    print(f"Epochs               : {args.epochs}")
    print(f"Val ratio            : {args.val_ratio}")
    print(f"Test ratio           : {args.test_ratio}")
    print(f"Output base          : {args.output}")
    print("=" * 80)

    rows = []
    for layer_name in layers:
        ckpt_path = _checkpoint_path(Path(args.output), args.model_name, layer_name, multi_run=multi_run)
        print(f"\n[RUN] model={args.model_name} layer={layer_name}")
        row = _run_single(
            model_name=args.model_name,
            dataset_roots=dataset_roots,
            output_path=ckpt_path,
            layer_name=layer_name,
            pooling=args.pooling,
            batch_size=args.batch_size,
            epochs=args.epochs,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lambda_bin=args.lambda_bin,
            lambda_cont=args.lambda_cont,
            seed=args.seed,
        )
        rows.append(row)

    if multi_run:
        summary_dir = Path(args.output).parent if Path(args.output).suffix == ".pt" else Path(args.output)
        csv_path, json_path = _write_sweep_summaries(rows, summary_dir)
        print(f"\nSweep summary CSV : {csv_path}")
        print(f"Sweep summary JSON: {json_path}")

        best = min(rows, key=lambda r: float(r["best_val_loss"]))
        print("\nBest layer:")
        print(f"  layer      : {best['layer_name']}")
        print(f"  val loss   : {best['best_val_loss']:.6f}")
        print(f"  checkpoint : {best['checkpoint_path']}")
    else:
        best = rows[0]
        print("\nTraining complete")
        print(f"Best epoch      : {best['best_epoch']}")
        print(f"Best val loss   : {best['best_val_loss']:.6f}")
        print(f"Train examples  : {best['num_train_examples']}")
        print(f"Val examples    : {best['num_val_examples']}")
        print(f"Test examples   : {best['num_test_examples']}")

    return 0


def _train_from_config(config_path: Path) -> int:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    shared = cfg.get("shared", {})
    models = cfg.get("models", [])

    if not models:
        raise ValueError("Config must contain a non-empty 'models' list")

    output_dir = Path(shared.get("output_dir", "artifacts/probe_sweeps"))

    rows: List[Dict] = []
    _print_header("Training Schwartz Probes (Config Sweep)")
    print(f"Config: {config_path}")
    print(f"Models: {len(models)}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    for model_cfg in models:
        model_name = str(model_cfg.get("name", "model")).strip()
        layers = [_normalize_layer_spec(x) for x in model_cfg.get("layers", [])]
        if not layers:
            raise ValueError(f"Model '{model_name}' has no layers listed")

        activations = model_cfg.get("activations", {})
        dataset_roots = {
            "valueeval": Path(activations["valueeval"]),
            "valuenet": Path(activations["valuenet"]),
            "fulcra": Path(activations["fulcra"]),
        }

        pooling = int(model_cfg.get("pooling", shared.get("pooling", 1)))
        batch_size = int(model_cfg.get("batch_size", shared.get("batch_size", 96)))
        epochs = int(model_cfg.get("epochs", shared.get("epochs", 10)))
        val_ratio = float(model_cfg.get("val_ratio", shared.get("val_ratio", 0.1)))
        test_ratio = float(model_cfg.get("test_ratio", shared.get("test_ratio", 0.1)))
        lr = float(model_cfg.get("lr", shared.get("lr", 1e-3)))
        weight_decay = float(model_cfg.get("weight_decay", shared.get("weight_decay", 1e-4)))
        lambda_bin = float(model_cfg.get("lambda_bin", shared.get("lambda_bin", 1.0)))
        lambda_cont = float(model_cfg.get("lambda_cont", shared.get("lambda_cont", 1.0)))
        seed = int(model_cfg.get("seed", shared.get("seed", 42)))

        print(f"\n[MODEL] {model_name}")
        print(f"  layers: {', '.join(layers)}")

        for layer_name in layers:
            ckpt_path = output_dir / f"schwartz_probe_{_safe_name(model_name)}_l{_layer_index(layer_name)}.pt"
            print(f"  [RUN] layer={layer_name}")
            row = _run_single(
                model_name=model_name,
                dataset_roots=dataset_roots,
                output_path=ckpt_path,
                layer_name=layer_name,
                pooling=pooling,
                batch_size=batch_size,
                epochs=epochs,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                lr=lr,
                weight_decay=weight_decay,
                lambda_bin=lambda_bin,
                lambda_cont=lambda_cont,
                seed=seed,
            )
            rows.append(row)

    csv_path, json_path = _write_sweep_summaries(rows, output_dir)
    print("\nSweep complete")
    print(f"Results CSV : {csv_path}")
    print(f"Results JSON: {json_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a 10D Schwartz probe (single layer, layer sweep, or config-driven multi-model sweep)"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON config for multi-model/multi-layer sweep",
    )
    parser.add_argument("--model-name", default="llama-3.1-8b-instruct", help="Model label used in output summaries")
    parser.add_argument("--layers", nargs="*", default=None, help="Candidate layers (e.g. 20 model.layers.31)")

    parser.add_argument(
        "--valuenet-activations",
        default="activations/valuenet",
        help="Activation directory for ValueNet examples",
    )
    parser.add_argument(
        "--valueeval-activations",
        default="activations/valueeval",
        help="Activation directory for ValueEval examples",
    )
    parser.add_argument(
        "--fulcra-activations",
        default="activations/fulcra",
        help="Activation directory for FULCRA examples",
    )
    parser.add_argument(
        "--output",
        default="artifacts/schwartz_probe.pt",
        help="Checkpoint path for single-layer mode; output base dir/file for sweep mode",
    )
    parser.add_argument(
        "--layer-name",
        default="model.layers.31",
        help="Transformer layer used for probe features (single-layer mode)",
    )
    parser.add_argument(
        "--pooling",
        type=int,
        default=1,
        help="Pooling over token axis: 1 last token (default), -1 full mean, k mean last-k",
    )
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-bin", type=float, default=1.0)
    parser.add_argument("--lambda-cont", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.config:
        return _train_from_config(Path(args.config))
    return _train_one_model_layer_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())

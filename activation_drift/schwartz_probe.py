import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import load_json, load_pickle


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

_VALUE_ALIASES = {
    "SELF DIRECTION": "SELF-DIRECTION",
    "SELF_DIRECTION": "SELF-DIRECTION",
    "SELF-DIRECTION": "SELF-DIRECTION",
    "STIMULATION": "STIMULATION",
    "HEDONISM": "HEDONISM",
    "ACHIEVEMENT": "ACHIEVEMENT",
    "POWER": "POWER",
    "SECURITY": "SECURITY",
    "TRADITION": "TRADITION",
    "CONFORMITY": "CONFORMITY",
    "BENEVOLENCE": "BENEVOLENCE",
    "UNIVERSALISM": "UNIVERSALISM",
}


def normalize_value_name(name: str) -> Optional[str]:
    key = str(name).strip().upper().replace("_", " ").replace("-", " ")
    key = re.sub(r"\s+", " ", key)
    return _VALUE_ALIASES.get(key)


def pool_activation(activation: np.ndarray, pooling: int) -> np.ndarray:
    if pooling == -1:
        return np.mean(activation, axis=(0, 1)).astype(np.float32)

    if pooling < 1:
        raise ValueError("pooling must be -1 or a positive integer")

    seq_len = activation.shape[1]
    k = min(pooling, seq_len)
    return np.mean(activation[:, -k:, :], axis=(0, 1)).astype(np.float32)


@dataclass
class ProbeExample:
    dataset_name: str
    activation_file: Path
    target: np.ndarray
    binary_mask: np.ndarray
    continuous_mask: np.ndarray


class LinearSchwartzProbe(nn.Module):
    def __init__(self, hidden_size: int, num_values: int = 10):
        super().__init__()
        self.head = nn.Linear(hidden_size, num_values)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(hidden)


class SchwartzProbeAxisAnalyzer:
    """Project current activations onto an axis defined by two Schwartz values."""

    def __init__(self, checkpoint_path: str | Path, value_a: str, value_b: str):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise ValueError(f"Probe checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.layer_name = checkpoint["layer_name"]
        self.pooling = int(checkpoint["pooling"])
        self.hidden_size = int(checkpoint["hidden_size"])
        self.schwartz_values = list(checkpoint.get("schwartz_values", SCHWARTZ_10_VALUES))

        self.value_a = self._resolve_value(value_a)
        self.value_b = self._resolve_value(value_b)
        self.value_a_index = self.schwartz_values.index(self.value_a)
        self.value_b_index = self.schwartz_values.index(self.value_b)

        self.model = LinearSchwartzProbe(hidden_size=self.hidden_size, num_values=len(self.schwartz_values))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def _compute_logits(self, current_activations: Dict[str, np.ndarray]) -> torch.Tensor:
        """Compute probe logits from current activations using checkpoint layer/pooling."""
        if self.layer_name not in current_activations:
            raise ValueError(f"Current activations do not contain layer {self.layer_name}")

        current_act = current_activations[self.layer_name]
        if hasattr(current_act, "detach"):
            current_act = current_act.detach().cpu().numpy()

        hidden = pool_activation(current_act, self.pooling)
        hidden_tensor = torch.tensor(hidden, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(hidden_tensor).squeeze(0)
        return logits

    def _resolve_value(self, value_name: str) -> str:
        normalized = normalize_value_name(value_name)
        if normalized is None:
            raise ValueError(
                f"Unknown Schwartz value '{value_name}'. Available values: {', '.join(self.schwartz_values)}"
            )
        if normalized not in self.schwartz_values:
            raise ValueError(
                f"Value '{normalized}' is not present in checkpoint labels: {', '.join(self.schwartz_values)}"
            )
        return normalized

    def predict_scores(self, current_activations: Dict[str, np.ndarray]) -> Dict[str, float]:
        logits = self._compute_logits(current_activations)
        probs = torch.sigmoid(logits)

        return {
            value_name: float(probs[idx].item())
            for idx, value_name in enumerate(self.schwartz_values)
        }

    def predict_logits(self, current_activations: Dict[str, np.ndarray]) -> Dict[str, float]:
        logits = self._compute_logits(current_activations)
        return {
            value_name: float(logits[idx].item())
            for idx, value_name in enumerate(self.schwartz_values)
        }

    def measure_activation_drift(self, current_activations: Dict[str, np.ndarray]) -> Dict[str, float]:
        logits = self._compute_logits(current_activations)

        axis_score = float(logits[self.value_a_index].item() - logits[self.value_b_index].item())
        return {self.layer_name: axis_score}


def _as_vector(label_obj, expected_len: int) -> np.ndarray:
    if isinstance(label_obj, dict):
        vec = np.zeros(expected_len, dtype=np.float32)
        for idx, value_name in enumerate(SCHWARTZ_10_VALUES):
            if value_name in label_obj:
                vec[idx] = float(label_obj[value_name])
        return vec

    arr = np.asarray(label_obj, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] != expected_len:
        raise ValueError(f"Expected a 1D vector of length {expected_len}, got shape {arr.shape}")
    return arr.astype(np.float32)


def _infer_label_type(vec: np.ndarray) -> str:
    is_binary = np.all(np.isin(vec, [0.0, 1.0]))
    return "binary" if is_binary else "continuous"


def _extract_targets_from_metadata(metadata: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(SCHWARTZ_10_VALUES)

    if "schwartz_10_binary" in metadata:
        target = _as_vector(metadata["schwartz_10_binary"], n)
        mask = _as_vector(metadata.get("schwartz_10_binary_mask", np.ones(n, dtype=np.float32)), n)
        return target, mask, np.zeros(n, dtype=np.float32)

    if "schwartz_10_continuous" in metadata:
        target = _as_vector(metadata["schwartz_10_continuous"], n)
        mask = _as_vector(metadata.get("schwartz_10_continuous_mask", np.ones(n, dtype=np.float32)), n)
        return target, np.zeros(n, dtype=np.float32), mask

    if "schwartz_10" in metadata:
        target = _as_vector(metadata["schwartz_10"], n)
        mask = _as_vector(metadata.get("schwartz_10_mask", np.ones(n, dtype=np.float32)), n)
        label_type = str(metadata.get("label_type", _infer_label_type(target))).lower()
        if label_type == "binary":
            return target, mask, np.zeros(n, dtype=np.float32)
        if label_type == "continuous":
            return target, np.zeros(n, dtype=np.float32), mask
        raise ValueError(f"Unsupported label_type '{label_type}'")

    behavior_type = str(metadata.get("behavior_type", "")).strip()
    match = re.match(r"^(.*)_(positive|negative)$", behavior_type, flags=re.IGNORECASE)
    if not match:
        raise ValueError("Metadata has no recognized Schwartz labels")

    value_name_raw, polarity = match.group(1), match.group(2).lower()
    value_name = normalize_value_name(value_name_raw)
    if value_name is None:
        raise ValueError(f"Unknown Schwartz value in behavior_type '{behavior_type}'")

    value_index = SCHWARTZ_10_VALUES.index(value_name)
    target = np.zeros(n, dtype=np.float32)
    target[value_index] = 1.0 if polarity == "positive" else 0.0
    binary_mask = np.zeros(n, dtype=np.float32)
    binary_mask[value_index] = 1.0

    return target, binary_mask, np.zeros(n, dtype=np.float32)


def discover_probe_examples(dataset_name: str, root_dir: Path) -> List[ProbeExample]:
    if not root_dir.exists():
        return []

    examples: List[ProbeExample] = []
    metadata_files = sorted(root_dir.rglob("metadata.json"))
    for metadata_file in metadata_files:
        example_dir = metadata_file.parent
        activation_file = example_dir / "activations.pkl"
        if not activation_file.exists():
            continue

        try:
            metadata = load_json(metadata_file)
            target, binary_mask, continuous_mask = _extract_targets_from_metadata(metadata)
            examples.append(
                ProbeExample(
                    dataset_name=dataset_name,
                    activation_file=activation_file,
                    target=target,
                    binary_mask=binary_mask,
                    continuous_mask=continuous_mask,
                )
            )
        except Exception:
            continue

    return examples


def _split_train_val_test(
    examples_by_dataset: Dict[str, List[ProbeExample]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[
    Dict[str, List[ProbeExample]],
    Dict[str, List[ProbeExample]],
    Dict[str, List[ProbeExample]],
]:
    if val_ratio < 0.0 or test_ratio < 0.0:
        raise ValueError("val_ratio and test_ratio must be non-negative")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    rng = np.random.default_rng(seed)

    train_split: Dict[str, List[ProbeExample]] = {}
    val_split: Dict[str, List[ProbeExample]] = {}
    test_split: Dict[str, List[ProbeExample]] = {}

    for ds_name, ds_examples in examples_by_dataset.items():
        if not ds_examples:
            train_split[ds_name] = []
            val_split[ds_name] = []
            test_split[ds_name] = []
            continue

        indices = np.arange(len(ds_examples))
        rng.shuffle(indices)

        n_total = len(indices)
        if n_total < 3:
            # Tiny datasets: keep everything in train to avoid empty/invalid splits.
            train_split[ds_name] = list(ds_examples)
            val_split[ds_name] = []
            test_split[ds_name] = []
            continue

        n_val = int(round(n_total * val_ratio))
        n_test = int(round(n_total * test_ratio))

        # Ensure all three splits are represented when dataset size permits.
        if n_total >= 10:
            n_val = max(1, n_val)
            n_test = max(1, n_test)

        # Leave at least one train sample.
        while n_val + n_test >= n_total:
            if n_val >= n_test and n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
            else:
                break

        test_idx = set(indices[:n_test].tolist())
        val_idx = set(indices[n_test:n_test + n_val].tolist())
        train_examples = [ex for i, ex in enumerate(ds_examples) if i not in test_idx and i not in val_idx]
        val_examples = [ex for i, ex in enumerate(ds_examples) if i in val_idx]
        test_examples = [ex for i, ex in enumerate(ds_examples) if i in test_idx]

        train_split[ds_name] = train_examples
        val_split[ds_name] = val_examples
        test_split[ds_name] = test_examples

    return train_split, val_split, test_split


def _safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path.resolve())


def _build_split_manifest(
    dataset_roots: Dict[str, Path],
    train_split: Dict[str, List[ProbeExample]],
    val_split: Dict[str, List[ProbeExample]],
    test_split: Dict[str, List[ProbeExample]],
) -> Dict[str, Dict[str, List[str]]]:
    manifest: Dict[str, Dict[str, List[str]]] = {}
    for ds_name, ds_root in dataset_roots.items():
        manifest[ds_name] = {
            "train": [_safe_relpath(ex.activation_file, ds_root) for ex in train_split.get(ds_name, [])],
            "val": [_safe_relpath(ex.activation_file, ds_root) for ex in val_split.get(ds_name, [])],
            "test": [_safe_relpath(ex.activation_file, ds_root) for ex in test_split.get(ds_name, [])],
        }
    return manifest


def _load_hidden_vector(example: ProbeExample, layer_name: str, pooling: int) -> np.ndarray:
    activations = load_pickle(example.activation_file)
    if layer_name not in activations:
        raise KeyError(f"Layer '{layer_name}' not found in {example.activation_file}")

    layer_activation = activations[layer_name]
    if hasattr(layer_activation, "detach"):
        layer_activation = layer_activation.detach().cpu().numpy()

    return pool_activation(layer_activation, pooling)


def _compute_pos_weight(train_examples_by_dataset: Dict[str, List[ProbeExample]]) -> torch.Tensor:
    pos = np.zeros(len(SCHWARTZ_10_VALUES), dtype=np.float64)
    neg = np.zeros(len(SCHWARTZ_10_VALUES), dtype=np.float64)

    for ds_examples in train_examples_by_dataset.values():
        for ex in ds_examples:
            bmask = ex.binary_mask > 0.5
            if not np.any(bmask):
                continue
            vals = ex.target[bmask]
            pos[bmask] += (vals > 0.5).astype(np.float64)
            neg[bmask] += (vals <= 0.5).astype(np.float64)

    weights = np.ones(len(SCHWARTZ_10_VALUES), dtype=np.float32)
    valid = pos > 0
    weights[valid] = (neg[valid] / pos[valid]).astype(np.float32)
    weights = np.clip(weights, 1.0, 20.0)

    return torch.tensor(weights, dtype=torch.float32)


def _build_balanced_batch(
    ds_names: List[str],
    train_examples_by_dataset: Dict[str, List[ProbeExample]],
    dataset_batch_size: int,
    layer_name: str,
    pooling: int,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_rows: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    b_masks: List[np.ndarray] = []
    c_masks: List[np.ndarray] = []
    dataset_ids: List[int] = []

    for ds_id, ds_name in enumerate(ds_names):
        ds_examples = train_examples_by_dataset[ds_name]
        if not ds_examples:
            continue

        replace = len(ds_examples) < dataset_batch_size
        idx = rng.choice(len(ds_examples), size=dataset_batch_size, replace=replace)
        for i in idx.tolist():
            ex = ds_examples[i]
            hidden_rows.append(_load_hidden_vector(ex, layer_name, pooling))
            targets.append(ex.target)
            b_masks.append(ex.binary_mask)
            c_masks.append(ex.continuous_mask)
            dataset_ids.append(ds_id)

    if not hidden_rows:
        raise ValueError("No training examples available to build a batch")

    hidden = torch.tensor(np.stack(hidden_rows), dtype=torch.float32)
    target = torch.tensor(np.stack(targets), dtype=torch.float32)
    binary_mask = torch.tensor(np.stack(b_masks), dtype=torch.float32)
    continuous_mask = torch.tensor(np.stack(c_masks), dtype=torch.float32)
    ds_ids = torch.tensor(dataset_ids, dtype=torch.long)
    return hidden, target, binary_mask, continuous_mask, ds_ids


def _compute_group_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    binary_mask: torch.Tensor,
    continuous_mask: torch.Tensor,
    pos_weight: torch.Tensor,
    lambda_bin: float,
    lambda_cont: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pos_weight)
    bden = binary_mask.sum().clamp_min(1.0)
    loss_bin = (bce * binary_mask).sum() / bden

    probs = torch.sigmoid(logits)
    huber = F.smooth_l1_loss(probs, target, reduction="none")
    cden = continuous_mask.sum().clamp_min(1.0)
    loss_cont = (huber * continuous_mask).sum() / cden

    loss = lambda_bin * loss_bin + lambda_cont * loss_cont
    return loss, loss_bin.detach(), loss_cont.detach()


def train_linear_schwartz_probe(
    dataset_roots: Dict[str, Path],
    output_path: Path,
    layer_name: str = "model.layers.31",
    pooling: int = 1,
    batch_size: int = 96,
    epochs: int = 10,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_bin: float = 1.0,
    lambda_cont: float = 1.0,
    seed: int = 42,
) -> Dict[str, float]:
    if batch_size < 3:
        raise ValueError("batch_size must be at least 3 for balanced 3-dataset training")

    examples_by_dataset: Dict[str, List[ProbeExample]] = {}
    for dataset_name, root in dataset_roots.items():
        examples_by_dataset[dataset_name] = discover_probe_examples(dataset_name, root)

    non_empty = [name for name, items in examples_by_dataset.items() if items]
    if not non_empty:
        raise ValueError("No probe examples found. Check activation directories and metadata labels")

    train_split, val_split, test_split = _split_train_val_test(
        examples_by_dataset,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    train_non_empty = [name for name in non_empty if train_split[name]]
    if not train_non_empty:
        raise ValueError("No training examples available after split")

    probe_input_example = train_split[train_non_empty[0]][0]
    hidden_size = _load_hidden_vector(probe_input_example, layer_name, pooling).shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearSchwartzProbe(hidden_size=hidden_size, num_values=len(SCHWARTZ_10_VALUES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_weight = _compute_pos_weight(train_split).to(device)

    dataset_batch_size = max(1, batch_size // len(train_non_empty))
    train_sizes = [len(train_split[name]) for name in train_non_empty]
    steps_per_epoch = max(1, math.ceil(max(train_sizes) / dataset_batch_size))
    rng = np.random.default_rng(seed)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for _ in range(steps_per_epoch):
            hidden, target, bmask, cmask, ds_ids = _build_balanced_batch(
                ds_names=train_non_empty,
                train_examples_by_dataset=train_split,
                dataset_batch_size=dataset_batch_size,
                layer_name=layer_name,
                pooling=pooling,
                rng=rng,
            )

            hidden = hidden.to(device)
            target = target.to(device)
            bmask = bmask.to(device)
            cmask = cmask.to(device)
            ds_ids = ds_ids.to(device)

            logits = model(hidden)
            group_losses = []

            for ds_id in ds_ids.unique().tolist():
                idx = ds_ids == ds_id
                g_loss, _, _ = _compute_group_loss(
                    logits=logits[idx],
                    target=target[idx],
                    binary_mask=bmask[idx],
                    continuous_mask=cmask[idx],
                    pos_weight=pos_weight,
                    lambda_bin=lambda_bin,
                    lambda_cont=lambda_cont,
                )
                group_losses.append(g_loss)

            loss = torch.stack(group_losses).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        val_loss = evaluate_linear_schwartz_probe(
            model=model,
            val_examples_by_dataset=val_split,
            layer_name=layer_name,
            pooling=pooling,
            pos_weight=pos_weight,
            lambda_bin=lambda_bin,
            lambda_cont=lambda_cont,
            device=device,
        )

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        print(
            f"Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state")

    split_manifest = _build_split_manifest(
        dataset_roots=dataset_roots,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state["model"],
            "best_epoch": best_state["epoch"],
            "best_val_loss": best_state["val_loss"],
            "layer_name": layer_name,
            "pooling": pooling,
            "hidden_size": hidden_size,
            "schwartz_values": SCHWARTZ_10_VALUES,
            "dataset_roots": {k: str(v) for k, v in dataset_roots.items()},
            "train_counts": {k: len(v) for k, v in train_split.items()},
            "val_counts": {k: len(v) for k, v in val_split.items()},
            "test_counts": {k: len(v) for k, v in test_split.items()},
            "split_manifest": split_manifest,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "split_seed": seed,
            "pos_weight": pos_weight.detach().cpu().numpy(),
            "lambda_bin": lambda_bin,
            "lambda_cont": lambda_cont,
        },
        output_path,
    )

    summary = {
        "best_val_loss": float(best_state["val_loss"]),
        "best_epoch": int(best_state["epoch"]),
        "hidden_size": int(hidden_size),
        "num_train_examples": int(sum(len(v) for v in train_split.values())),
        "num_val_examples": int(sum(len(v) for v in val_split.values())),
        "num_test_examples": int(sum(len(v) for v in test_split.values())),
    }
    return summary


@torch.no_grad()
def evaluate_linear_schwartz_probe(
    model: LinearSchwartzProbe,
    val_examples_by_dataset: Dict[str, List[ProbeExample]],
    layer_name: str,
    pooling: int,
    pos_weight: torch.Tensor,
    lambda_bin: float,
    lambda_cont: float,
    device: torch.device,
) -> float:
    model.eval()
    ds_losses = []

    for _, ds_examples in val_examples_by_dataset.items():
        if not ds_examples:
            continue

        hidden_rows = []
        targets = []
        b_masks = []
        c_masks = []
        for ex in ds_examples:
            try:
                hidden_rows.append(_load_hidden_vector(ex, layer_name, pooling))
                targets.append(ex.target)
                b_masks.append(ex.binary_mask)
                c_masks.append(ex.continuous_mask)
            except Exception:
                continue

        if not hidden_rows:
            continue

        hidden = torch.tensor(np.stack(hidden_rows), dtype=torch.float32, device=device)
        target = torch.tensor(np.stack(targets), dtype=torch.float32, device=device)
        bmask = torch.tensor(np.stack(b_masks), dtype=torch.float32, device=device)
        cmask = torch.tensor(np.stack(c_masks), dtype=torch.float32, device=device)

        logits = model(hidden)
        loss, _, _ = _compute_group_loss(
            logits=logits,
            target=target,
            binary_mask=bmask,
            continuous_mask=cmask,
            pos_weight=pos_weight,
            lambda_bin=lambda_bin,
            lambda_cont=lambda_cont,
        )
        ds_losses.append(float(loss.detach().cpu()))

    if not ds_losses:
        return float("inf")
    return float(np.mean(ds_losses))
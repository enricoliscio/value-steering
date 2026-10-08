import json
import pickle
import random
import time
from datetime import datetime
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from ollama import ResponseError

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None


DEFAULT_LLM_MODEL_PATH = "/home/USER/dev/models/meta-llama/Llama-3.1-8B-Instruct"


def save_json(data, filepath):
    """Save data to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(filepath):
    """Load data from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def save_pickle(data, filepath):
    """Save data to a pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filepath):
    """Load data from a pickle file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def load_model(model_path, device, quantization: str = "none"):
    quantization = str(quantization).lower().strip()
    if quantization not in {"none", "8bit", "4bit"}:
        raise ValueError("quantization must be one of: none, 8bit, 4bit")

    if quantization != "none" and device != "cuda":
        print("Warning: quantization requested without CUDA; falling back to non-quantized load")
        quantization = "none"

    if quantization != "none" and BitsAndBytesConfig is None:
        raise ImportError(
            "Quantization requires transformers BitsAndBytesConfig support and bitsandbytes. "
            "Install bitsandbytes in the active environment."
        )

    model_kwargs = {
        "torch_dtype": "auto" if device == "cuda" else torch.float32,
        "device_map": "auto" if device == "cuda" else None,
        "low_cpu_mem_usage": True,
    }

    if quantization == "8bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()

    # Report placement details so mixed CPU/CUDA offload is visible in logs.
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict) and hf_device_map:
        counts = {}
        for target in hf_device_map.values():
            label = str(target)
            counts[label] = counts.get(label, 0) + 1
        summary = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        print(f"HF device map: {summary}")

        cpu_modules = [name for name, target in hf_device_map.items() if str(target) == "cpu"]
        if cpu_modules:
            sample = ", ".join(cpu_modules[:6])
            if len(cpu_modules) > 6:
                sample += ", ..."
            print(f"HF offload warning: {len(cpu_modules)} modules on CPU ({sample})")

    return model

def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_path(filepath):
    return Path(filepath)

def make_directory(path):
    if isinstance(path, str):
        path = get_path(path)
    path.mkdir(parents=True, exist_ok=True)


def require_existing_path(pathlike, kind: str = "any") -> Path:
    """Return a validated Path or raise FileNotFoundError.

    Args:
        pathlike: Path-like value to validate.
        kind: one of "any", "file", "dir".
    """
    path = Path(pathlike)
    if kind == "file":
        ok = path.is_file()
    elif kind == "dir":
        ok = path.is_dir()
    else:
        ok = path.exists()

    if not ok:
        if kind == "file":
            expected = "file"
        elif kind == "dir":
            expected = "directory"
        else:
            expected = "path"
        raise FileNotFoundError(f"Expected existing {expected}: {path}")
    return path


def infer_targets_from_condition(
    condition: str,
    value_a: str,
    value_b: str,
) -> tuple[str, str]:
    """Infer (model_target_value, user_target_value) from a condition name."""
    cond = str(condition).strip().lower()
    if cond == "baseline":
        return "", ""
    if cond in {"user_va", "user_value1"}:
        return "", value_a
    if cond in {"user_vb", "user_value2"}:
        return "", value_b
    if cond in {"both_va", "both_value1"}:
        return value_a, value_a
    if cond in {"both_vb", "both_value2"}:
        return value_b, value_b
    if cond in {"model_vb_user_va", "model_v2_user_v1"}:
        return value_b, value_a
    if cond in {"model_va_user_vb", "model_v1_user_v2"}:
        return value_a, value_b
    else:
        raise ValueError(
            "User and target model value target were not specified, and I could not "
            f"guess them from the condition name: {cond}"
        )


def ollama_chat_with_retries(
    client,
    model: str,
    messages,
    options: dict,
    max_retries: int = 10,
    base_delay: float = 1.0,
) -> str:
    """Call Ollama chat with retries, matching original exp_setup behavior."""
    for attempt in range(max_retries):
        try:
            chunks = []
            for part in client.chat(model=model, messages=messages, stream=True, options=options):
                chunk = str(part.get("message", {}).get("content", ""))
                if chunk:
                    chunks.append(chunk)

            text = "".join(chunks).strip()
            text = text.split("\n")[0].strip()
            if text:
                return text

            # Rarely, streaming can complete without yielding usable text.
            if attempt == max_retries - 1:
                return ""

            delay = base_delay * (2 ** attempt)
            delay += random.uniform(0, 0.5)
            print(f"Attempt {attempt + 1} returned empty output, retrying in {delay:.1f}s...")
            time.sleep(delay)
            continue

        except ResponseError as exc:
            status = getattr(exc, "status_code", None)
            retryable = status in (429, 500, 502, 503, 504) or "overloaded" in str(exc).lower()
            if (not retryable) or attempt == max_retries - 1:
                raise

            delay = base_delay * (2 ** attempt)
            delay += random.uniform(0, 0.5)

            print(f"Attempt {attempt + 1} failed ({exc}), retrying in {delay:.1f}s...")

            time.sleep(delay)

    raise RuntimeError("Unreachable retry state")
import numpy as np
from typing import Dict, List, Optional
import pandas as pd
import torch

from .utils import *
from .probe import ActivationProbe


class SteeringVectorAnalyzer:
    """Computes and analyzes steering vectors between different behaviors."""

    def __init__(self, activation_base_dir: str = "behavior_activations"):
        """
        Initialize the steering vector analyzer.
        
        Args:
            activation_base_dir: Base directory containing saved activations
        """
        self.activation_base_dir = get_path(activation_base_dir)
        self.steering_vectors = {}
        # Pooling semantics:
        # -1: mean over full sequence, 1: last token only, k>1: mean over last k tokens
        self.pooling = 1

    def _pool_activation(self, activation: np.ndarray, pooling: int) -> np.ndarray:
        """Pool activation with configurable sequence strategy.

        Args:
            activation: Activation tensor with shape (1, seq_len, hidden_size)
            pooling: -1 for full mean, 1 for last token, k for mean over last k

        Returns:
            Vector of shape (hidden_size,)
        """
        if pooling == -1:
            return np.mean(activation, axis=(0, 1)).astype(np.float32)

        if pooling < 1:
            raise ValueError("pooling must be -1 or a positive integer")

        # Clamp k to available sequence length so short prompts still work.
        seq_len = activation.shape[1]
        k = min(pooling, seq_len)
        return np.mean(activation[:, -k:, :], axis=(0, 1)).astype(np.float32)

    def compute_steering_vector(self, behavior1: str, behavior2: str, pooling: int = 1) -> Dict[str, np.ndarray]:
        """
        Compute steering vector as the mean difference between two behaviors.
        
        Args:
            behavior1: First behavior type directory
            behavior2: Second behavior type directory (direction will be behavior1 - behavior2)
            pooling: 1 last token (default), -1 full-sequence mean, k mean over last k tokens
        
        Returns:
            Dictionary mapping layer names to steering vectors (numpy arrays)
        """
        print(f"Computing steering vector: {behavior1} -> {behavior2} (pooling={pooling})")
        self.pooling = pooling

        # Load all activations for each behavior
        activations1 = self._load_behavior_activations(behavior1)
        activations2 = self._load_behavior_activations(behavior2)

        if not activations1 or not activations2:
            raise ValueError(f"Could not load activations for {behavior1} or {behavior2}")

        print(f"  {behavior1}: {len(activations1)} examples")
        print(f"  {behavior2}: {len(activations2)} examples")

        steering_vectors = {}

        # Get common layers
        common_layers = set(activations1[0].keys()) & set(activations2[0].keys())

        for layer in sorted(common_layers):
            # Pool each example using the configured sequence strategy.
            means1 = np.array([self._pool_activation(act[layer], pooling) for act in activations1])
            means2 = np.array([self._pool_activation(act[layer], pooling) for act in activations2])

            # Average across examples
            mean1 = np.mean(means1, axis=0)  # (hidden_size,)
            mean2 = np.mean(means2, axis=0)  # (hidden_size,)

            # Compute steering vector as difference
            steering = (mean1 - mean2).astype(np.float32)
            steering_vectors[layer] = steering

        print(f"✓ Computed {len(steering_vectors)} steering vectors")
        self.steering_vectors = steering_vectors
        return steering_vectors

    def _load_behavior_activations(self, behavior: str) -> List[Dict[str, np.ndarray]]:
        """Load all activation files for a behavior."""
        behavior_dir = self.activation_base_dir / behavior
        if not behavior_dir.exists():
            print(f"Warning: Behavior directory '{behavior}' not found")
            return []

        activations = []
        example_dirs = sorted(behavior_dir.glob("*/"))

        for ex_dir in example_dirs:
            try:
                activation_file = ex_dir / "activations.pkl"
                activations.append(load_pickle(activation_file))
            except Exception as e:
                print(f"  Warning: Could not load {ex_dir}: {e}")

        return activations

    def measure_activation_drift(self, current_activations: Dict[str, np.ndarray], pooling: Optional[int] = None) -> Dict[str, float]:
        """
        Measure dot product between current activations and steering vectors.
        
        Args:
            current_activations: Dictionary of current activations (from a forward pass)
            pooling: Optional override. If None, uses the pooling used for steering vectors.
        
        Returns:
            Dictionary mapping layer names to dot products
        """
        if not self.steering_vectors:
            raise ValueError("Must call compute_steering_vector first")

        effective_pooling = self.pooling if pooling is None else pooling

        drift_scores = {}

        for layer, steering in self.steering_vectors.items():
            if layer not in current_activations:
                continue

            # Get current activation and convert to numpy if needed
            current_act = current_activations[layer]  # shape: (1, seq_len, hidden_size)
            
            # Handle both torch tensors and numpy arrays
            if hasattr(current_act, 'detach'):  # It's a torch tensor
                current_act = current_act.detach().cpu().numpy()
            
            # Pool with the same strategy used by steering vectors (unless overridden).
            current_avg = self._pool_activation(current_act, effective_pooling)

            # Compute dot product
            dot_product = np.dot(current_avg, steering)
            drift_scores[layer] = float(dot_product)

        return drift_scores

    def save_steering_vectors(self, filepath: str):
        """Save steering vectors to file."""
        steering_dict = {layer: vec.tolist() for layer, vec in self.steering_vectors.items()}
        save_json(steering_dict, filepath)

    def load_steering_vectors(self, filepath: str):
        """Load steering vectors from file."""
        steering_dict = load_json(filepath)
        self.steering_vectors = {layer: np.array(vec, dtype=np.float32) for layer, vec in steering_dict.items()}
        print(f"✓ Loaded {len(self.steering_vectors)} steering vectors from {filepath}")


class BehaviorActivationProcessor:
    """Processes CSV files with behavior examples and extracts activations."""

    def __init__(self, model_path: str, quantization: str = "none", output_dir: str = "behavior_activations", save_last_token_only: bool = True):
        """
        Initialize the behavior processor.
        
        Args:
            model_path: Path to the Llama model directory
            output_dir: Directory where extracted activations are stored
        """
        self.device = get_device()
        print(f"Loading model from {model_path} for batch processing...")

        # Load tokenizer and model
        self.tokenizer = load_tokenizer(model_path)
        self.model = load_model(model_path, self.device, quantization=quantization)

        # Initialize activation probe
        self.probe = ActivationProbe(self.model)
        self.save_last_token_only = bool(save_last_token_only)

        # Create output directory for activations
        self.activation_dir = get_path(output_dir)
        make_directory(self.activation_dir)

        print(f"✓ Model loaded on {self.device}")
        print("✓ Activation probe initialized")

    def process_csv(self, csv_path: str, text_column: str = "text", 
                   behavior_column: Optional[str] = "behavior", 
                   max_length: int = 512) -> Dict[str, List[str]]:
        """
        Process a CSV file with behavior examples.
        
        Args:
            csv_path: Path to CSV file
            text_column: Name of column containing text
            behavior_column: Name of column containing behavior type (optional)
            max_length: Maximum token length for each example
        
        Returns:
            Dictionary mapping behavior types to lists of processed example IDs
        """
        print(f"Reading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)

        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in CSV")

        # Group by behavior if specified
        if behavior_column and behavior_column in df.columns:
            behavior_groups = df.groupby(behavior_column)
            print(f"Found {len(behavior_groups)} behavior types: {list(behavior_groups.groups.keys())}")
        else:
            # Treat all as single behavior type
            behavior_groups = [("all", df)]
            print("No behavior column specified, processing all examples together")

        processed_examples = {}

        for behavior_type, group_df in behavior_groups:
            print(f"\nProcessing {len(group_df)} examples for behavior: {behavior_type}")
            example_ids = []

            for idx, row in group_df.iterrows():
                text = str(row[text_column]).strip()
                if not text:
                    continue

                example_id = f"{behavior_type}_{idx}"
                print(f"  Processing example {idx+1}/{len(group_df)}: {text[:50]}...")

                # Process the example
                success = self._process_single_example(text, example_id, behavior_type, max_length)
                if success:
                    example_ids.append(example_id)

            processed_examples[behavior_type] = example_ids
            print(f"✓ Completed {len(example_ids)} examples for {behavior_type}")

        return processed_examples

    def _process_single_example(self, text: str, example_id: str,
                               behavior_type: str, max_length: int,
                               extra_metadata: Optional[Dict] = None) -> bool:
        """
        Process a single text example and save its activations.
        
        Args:
            text: The text to process
            example_id: Unique identifier for this example
            behavior_type: Type of behavior this example represents
            max_length: Maximum token length
        
        Returns:
            True if processing was successful
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_length,
                padding=True
            ).to(self.device)

            # Clear previous activations
            self.probe.clear()

            # Forward pass to get activations
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Save activations and metadata
            self._save_example_data(
                text,
                example_id,
                behavior_type,
                inputs,
                extra_metadata=extra_metadata,
            )

            return True

        except Exception as e:
            print(f"    Error processing example {example_id}: {e}")
            return False

    def _save_example_data(self, text: str, example_id: str,
                          behavior_type: str, inputs: Dict[str, torch.Tensor],
                          extra_metadata: Optional[Dict] = None):
        """Save activations and metadata for a single example."""
        timestamp = get_timestamp()

        # Create behavior-specific directory
        behavior_dir = self.activation_dir / behavior_type
        make_directory(behavior_dir)

        # Create example directory
        example_dir = behavior_dir / f"{example_id}_{timestamp}"
        make_directory(example_dir)

        # Save activations
        activation_file = example_dir / "activations.pkl"
        self.probe.save_activations(str(activation_file), last_token_only=self.save_last_token_only)

        # Save metadata
        metadata = {
            "example_id": example_id,
            "behavior_type": behavior_type,
            "timestamp": timestamp,
            "text": text,
            "input_length": inputs['input_ids'].shape[1],
            "num_layers": len(self.probe.activations),
            "layer_names": list(self.probe.activations.keys()),
            "activation_storage": "last_token_only" if self.save_last_token_only else "full_sequence",
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        metadata_file = example_dir / "metadata.json"
        save_json(metadata, metadata_file)

        print(f"    ✓ Saved to {example_dir}")

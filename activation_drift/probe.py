import torch

from .utils import save_pickle


class ActivationProbe:
    """Captures and stores activations from model layers."""

    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on main transformer layers only."""
        def make_hook(name):
            def hook(_, __, output):
                # For LlamaDecoderLayer, capture the hidden states after the layer
                if hasattr(output, 'last_hidden_state'):
                    self.activations[name] = output.last_hidden_state.detach().cpu().float()
                elif isinstance(output, torch.Tensor):
                    self.activations[name] = output.detach().cpu().float()
                elif isinstance(output, tuple) and len(output) > 0:
                    if isinstance(output[0], torch.Tensor):
                        self.activations[name] = output[0].detach().cpu().float()
            return hook

        # Only register hooks on main transformer layers (model.layers.X)
        for name, module in self.model.named_modules():
            if name.startswith('model.layers.') and name.count('.') == 2:
                # This is a main decoder layer
                hook = module.register_forward_hook(make_hook(name))
                self.hooks.append(hook)

    def clear(self):
        """Clear stored activations."""
        self.activations = {}

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def save_activations(self, filepath: str, last_token_only: bool = False):
        """Save activations to file.

        Args:
            filepath: Destination pickle path.
            last_token_only: If True, keep only the final sequence position for
                each saved layer activation. The resulting tensors keep a
                sequence dimension of length 1 so downstream loaders continue to
                see a 3D shape.
        """
        activations_np = {}
        for name, tensor in self.activations.items():
            tensor_to_save = tensor[:, -1:, :] if last_token_only and tensor.ndim == 3 else tensor
            activations_np[name] = tensor_to_save.numpy()

        save_pickle(activations_np, filepath)
        print(f"✓ Activations saved to {filepath}")

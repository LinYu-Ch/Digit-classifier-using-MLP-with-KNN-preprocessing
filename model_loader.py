"""Loads your ConvNetwork and runs inference.

This mirrors your existing script: a state_dict loaded into ConvNetwork, raw
[0, 1] pixels with no Normalize, and softmax applied to the model's logits.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from model import ConvNetwork

NUM_CLASSES = 10

# Your inference script goes straight from ToTensor() to the model, so pixels stay
# in [0, 1] and nothing else is applied. If your TRAINING script had a
# T.Normalize((0.1307,), (0.3081,)) that inference was missing, set this to True --
# a mismatch here is a common cause of confident wrong answers.
NORMALIZE = False
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

_model: ConvNetwork | None = None
_device: torch.device | None = None


def load_model(model_path: str | None = None, device: str | None = None) -> str:
    """Load weights once at startup. Returns a label for /health."""
    global _model, _device

    path = Path(model_path or os.getenv("MODEL_PATH") or "cnn_mnist_weights.pth")
    device = device or os.getenv("MODEL_DEVICE")
    if not path.exists():
        raise FileNotFoundError(
            f"No weights at {path.resolve()}. Pass --model /path/to/cnn_mnist_weights.pth"
        )

    _device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = ConvNetwork(num_classes=NUM_CLASSES)
    state = torch.load(path, map_location=_device, weights_only=True)
    model.load_state_dict(state)
    model.to(_device)
    model.eval()

    _model = model
    return f"torch · {_device.type}"


@torch.no_grad()
def predict_probs(x: np.ndarray) -> np.ndarray:
    """(28, 28) float32 in [0, 1], ink already white-on-black -> (10,) probabilities."""
    if _model is None:
        raise RuntimeError("load_model() has not run.")

    arr = (x - MNIST_MEAN) / MNIST_STD if NORMALIZE else x

    # (28, 28) -> (1, 1, 28, 28), the shape your dataloader produced.
    tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0).to(_device)

    logits = _model(tensor)
    probs = torch.softmax(logits, dim=1)
    return probs.squeeze(0).cpu().numpy()


def backend_name() -> str:
    return "not loaded" if _model is None else f"torch · {_device.type}"

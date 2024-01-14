import numpy as np


def H(x: np.ndarray) -> np.ndarray:
    """The Heaviside (threshold) function."""
    return (x > 0.0).astype(np.int_)


def saturated_sigmoid(x: np.ndarray) -> np.ndarray:
    """The saturated sigmoid function."""
    return np.clip(x, 0, 1)


def ReLU(x: np.ndarray) -> np.ndarray:
    """The ReLU function."""
    return np.maximum(x, 0)

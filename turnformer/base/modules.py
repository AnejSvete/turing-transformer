from typing import Sequence, Tuple

import numpy as np


def construct_and(D: int, idx: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Constructs the AND gate in the form of a MLP.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The weights and biases of the AND gate.
    """
    W = np.zeros((1, D))
    W[0, idx] = 1
    b = np.asarray([-(len(idx) - 1)])
    return W, b

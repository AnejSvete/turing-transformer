import numpy as np


class ProjectionFunctions:
    @staticmethod
    def averaging_hard_normalization(s: np.ndarray) -> np.ndarray:
        """Applies the projection function of averaging hard attention mechanism
            to the input scores.

        Args:
            s (np.ndarray): The input scores.

        Returns:
            np.ndarray: The output scores.
        """
        a = s == np.max(s)
        return np.asarray(a, dtype=np.float32) / np.sum(a)

    @staticmethod
    def unique_hard_normalization(s: np.ndarray) -> np.ndarray:
        """Applies the projection function of unique hard attention mechanism
            to the input scores.

        Args:
            s (np.ndarray): The input scores.

        Returns:
            np.ndarray: The output scores.
        """
        a = np.zeros_like(s)
        a[np.argmax(s)] = 1
        return a

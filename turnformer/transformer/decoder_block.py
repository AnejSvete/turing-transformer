from typing import Tuple, Optional, Callable

from sympy import Rational
import sympy as sp

from turnformer.transformer.symbol_encoding import Embedding
from turnformer.transformer.attention import attention


class DecoderBlock:
    """Implementation of the skeleton of a single decoder block."""

    def __init__(self, D: int) -> None:
        self.D = D

        self.K = sp.zeros(self.D, dtype=Rational)
        self.V = sp.zeros(self.D, dtype=Rational)
        # self.b_k = sp.zeros(self.D, dtype=Rational)
        # self.b_v = sp.zeros(self.D, dtype=Rational)
        self.O = None  # type: Optional[Callable[[sp.Matrix], sp.Matrix]]

    def __call__(
        self, X: sp.Matrix, K_e: sp.Matrix, V_e: sp.Matrix
    ) -> Tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
        K, V = X @ self.K.T, X @ self.V.T

        X = attention(Q=X, K=K, V=V) + X

        X = attention(Q=X, K=K_e, V=V_e) + X

        if self.O is not None:
            for n in range(X.shape[0]):
                X[n, :] = self.O(X[n, :])

        return X, K, V

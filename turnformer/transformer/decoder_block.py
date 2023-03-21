from typing import Tuple, Optional, Callable

from sympy import Rational
import sympy as sp

from turnformer.transformer.symbol_embedding import Embedding
from turnformer.transformer.attention import attention


class DecoderBlock:
    """Implementation of the skeleton of a single decoder block."""

    def __init__(self, D: int) -> None:
        self.D = D

        self.Q = sp.zeros(self.D, dtype=Rational)
        self.b_q = sp.zeros(self.D, 1, dtype=Rational)
        self.K = sp.zeros(self.D, dtype=Rational)
        self.b_k = sp.zeros(self.D, 1, dtype=Rational)
        self.V = sp.zeros(self.D, dtype=Rational)
        self.b_v = sp.zeros(self.D, 1, dtype=Rational)
        self.O = None  # type: Optional[Callable[[sp.Matrix], sp.Matrix]]

    def __call__(self, X: sp.Matrix, K_e: sp.Matrix, V_e: sp.Matrix) -> sp.Matrix:
        Q = X @ self.Q.T
        K = X @ self.K.T
        V = X @ self.V.T

        P = attention(Q=Q, K=K, V=V) + Q

        A = attention(Q=P, K=K_e, V=V_e) + P

        if self.O is not None:
            for n in range(A.shape[0]):
                A[n, :] = self.O(A[n, :])

        return A

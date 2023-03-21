from typing import Tuple

from sympy import Rational
import sympy as sp

from turnformer.transformer.symbol_embedding import Embedding


class EncoderBlock:
    """Implementation of the encoder from the paper.
    It simply transforms the input embeddings into the key and value outputs of the
    encoder by using no self-attention and then linearly transforming the inputs into
    the output keys and values.
    """

    def __init__(self, embedding: Embedding) -> None:
        self.embedding = embedding

        # The key and value matrices for transforming the inputs are zero
        # to get the identity function in the attention.
        self.K1 = sp.zeros(self.embedding.D, dtype=Rational)
        self.V1 = sp.zeros(self.embedding.D, dtype=Rational)

        # The key and value matrices for transforming the outputs
        self.construct_K_transform()
        self.construct_V_transform()

    def construct_K_transform(self) -> None:
        """Constructs the key matrix for transforming the input encodings
        into the final key matrix.
        """
        self.b2_k = sp.zeros(self.embedding.D, 1, dtype=Rational)
        self.b2_k[0, self.embedding.offset(component=4) + 1] = Rational(-1)

        self.K2 = sp.zeros(self.embedding.D, dtype=Rational)
        # Copy the position value `n` from the input encoding.
        self.K2[
            self.embedding.offset(component=4), self.embedding.offset(component=4) + 1
        ] = Rational(1)

    def construct_V_transform(self) -> None:
        """Constructs the key matrix for transforming the input encodings
        into the final value matrix.
        """
        self.b2_v = sp.zeros(self.embedding.D, 1, dtype=Rational)

        self.V2 = sp.zeros(self.embedding.D, dtype=Rational)
        # Copy the one-hot encoding of the symbol from the input encoding.
        self.V2[
            self.embedding.offset(component=3) : self.embedding.offset(component=3)
            + len(self.embedding.Γ),
            self.embedding.offset(component=3) : self.embedding.offset(component=3)
            + len(self.embedding.Γ),
        ] = sp.eye(len(self.embedding.Γ), dtype=Rational)
        # Copy the position value `n` from the input encoding.
        self.K2[
            self.embedding.offset(component=3) + len(self.embedding.Γ),
            self.embedding.offset(component=4) + 1,
        ] = Rational(1)

    def __call__(self, X: sp.Matrix) -> Tuple[sp.Matrix, sp.Matrix]:
        """Transforms the input embeddings into the key and value outputs of the
        encoder by using no self-attention and then linearly transforming the inputs
        into the output keys and values.
        """
        # Transform the input embeddings into the key and value outputs of the
        # encoder by using no self-attention.

        # Linearly transform the inputs into the output keys and values.
        K = self.K2 @ X.T + self.b2_k
        V = self.V2 @ X.T + self.b2_v

        return K, V

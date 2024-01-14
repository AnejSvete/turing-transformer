from typing import Tuple
from itertools import product

from sympy import Rational
import sympy as sp

from turnformer.base.symbol import EMPTY, Sym
from turnformer.base.string import String
from turnformer.base.turing_machine import Direction
from turnformer.transformer.symbol_embedding import Embedding
from turnformer.transformer.attention import attention
from turnformer.transformer.decoder_block import DecoderBlock

sym_x = sp.Symbol("x")
# This is the saturated sigmoid function
σ = sp.Piecewise(
    (Rational(0), sym_x <= 0),
    (Rational(1), sym_x >= Rational(1)),
    (sp.Abs(sym_x) <= sp.sympify(1)),
)


def replication_matrix(n: int, N: int) -> sp.Matrix:
    R = sp.zeros(N * n, n, dtype=Rational)
    for i in range(N):
        R[i * n : (i + 1) * n, :] = sp.eye(n, dtype=Rational)
    return R


def m_hat(m: Direction) -> Rational:
    return Rational(f"1/2*{int(m)}+1/2")


def one_hot(n: int, N: int) -> sp.Matrix:
    """Returns the one-hot vector of length `N` with a 1 at position `n`."""
    v = sp.zeros(N, 1, dtype=Rational)
    v[n, 0] = Rational(1)
    return v


class Decoder:
    def __init__(self, embedding: Embedding) -> None:
        self.embedding = embedding
        self.D = self.embedding.D
        self.Q = self.embedding.Q
        self.Γ = self.embedding.Γ
        self.T = self.embedding.T

        self.decoder_block1 = DecoderBlock(self.embedding.D)
        self.setup_block_1()

        self.decoder_block2 = DecoderBlock(self.embedding.D)
        self.decoder_block3 = DecoderBlock(self.embedding.D)

        self.P_h_state = self.construct_state_projection_matrix()
        self.P_h_symbol = self.construct_symbol_projection_matrix()
        self.P_h_direction = self.construct_direction_projection_matrix()

        self.R_state = replication_matrix(len(self.Q), len(self.Γ))
        self.R_symbol = replication_matrix(len(self.Γ), len(self.Q))

    def S_n(self, n: int) -> sp.Matrix:
        """Returns the matrix `S_n`."""
        S_n = sp.zeros(len(self.Q), len(self.Γ), dtype=Rational)
        S_n[:, n] = sp.ones(len(self.Γ), 1, dtype=Rational)
        return S_n

    def construct_state_projection_matrix(self) -> sp.Matrix:
        """Returns the matrix `P`."""
        P = sp.zeros(len(self.Q), self.D, dtype=Rational)
        P[:, : len(self.Q)] = sp.eye(len(self.Q), dtype=Rational)
        return P

    def construct_symbol_projection_matrix(self) -> sp.Matrix:
        """Returns the matrix `P`."""
        P = sp.zeros(len(self.Γ), self.D, dtype=Rational)
        P[:, len(self.Q) : len(self.Q) + len(self.Γ)] = sp.eye(
            len(self.Γ), dtype=Rational
        )
        return P

    def construct_direction_projection_matrix(self) -> sp.Matrix:
        """Returns the matrix `P`."""
        P = sp.zeros(1, self.D, dtype=Rational)
        P[:, len(self.Q) + len(self.Γ)] = 1
        return P

    def construct_S_matrix(self) -> Tuple[sp.Matrix, sp.Matrix]:
        S = sp.zeros(len(self.Γ) * len(self.Q), len(self.Γ), dtype=Rational)
        b = Rational(-1) * sp.ones(len(self.Γ) * len(self.Q), 1, dtype=Rational)

        for n in range(len(self.Γ)):
            S_n = sp.zeros(len(self.Q), len(self.Γ), dtype=Rational)
            S_n[:, n] = sp.ones(len(self.Γ), 1, dtype=Rational)
            S[n * len(self.Q) : (n + 1) * len(self.Q), :] = S_n

        return S, b

    def construct_transition_matrix(self) -> None:
        """Constructs the transition matrix `T` encoding the transitions in the Turing
        machine.

        The transition matrix is a matrix of size `|Q| * |Γ|` by `2 * |Q| * |Γ|`.

        `M_{[q, s], [p, r, m]} = Rational(1)` iff the Turing machine transitions from state
        `q` to state `p` and writes symbol `r` to the tape and moves the head in
        direction `m` when reading symbol `s` from the tape.

        """
        T = sp.zeros(
            2 * len(self.Q) * len(self.Γ),
            len(self.Q) * len(self.Γ),
            dtype=Rational,
        )

        for q in self.Q:
            for s in self.Γ:
                for _, t, p, r, m in self.embedding.T.transitions(q):
                    if t != s:
                        continue

                    # The transition matrix encodes the transitions in the Turing
                    # machine.
                    T[
                        one_hot(
                            self.T.qsymd2idx[p, r, m], len(self.T.Q) * len(self.T.Γ) * 2
                        ),
                        one_hot(self.T.qsym2idx[q, s], len(self.T.Q) * len(self.T.Γ)),
                    ] = Rational(1)

    def construct_separation_matrix(self) -> sp.Matrix:
        A = sp.zeros(
            len(self.Q) + len(self.Γ) + 1, len(self.Q) * len(self.Γ) * 2, dtype=Rational
        )

        for q, s, m in product(self.Q, self.Γ, [Direction.LEFT, Direction.RIGHT]):
            A[
                self.T.q2idx[q],
                self.T.qsymd2idx[q, s, m],
            ] = Rational(1)
            A[
                len(self.Q) + self.T.sym2idx[s],
                self.T.qsymd2idx[q, s, m],
            ] = Rational(1)
            A[
                len(self.Q) + len(self.Γ),
                self.T.qsymd2idx[q, s, m],
            ] = Rational(m)

        return A

    def construct_h_1_parameters(self) -> Tuple[sp.Matrix, sp.Matrix]:
        H = sp.zeros(
            len(self.Q) + len(self.Γ) + 1 + len(self.Q) * len(self.Γ),
            self.D,
            dtype=Rational,
        )
        b = sp.zeros(
            len(self.Q) + len(self.Γ) + 1 + len(self.Q) * len(self.Γ), 1, dtype=Rational
        )

        H[: len(self.Q) + len(self.Γ), : len(self.Q) + len(self.Γ)] = sp.eye(
            len(self.Q) + len(self.Γ), dtype=Rational
        )
        H[len(self.Q) + len(self.Γ), len(self.Q) + len(self.Γ)] = Rational("1/2")
        S, b_prime = self.construct_S_matrix()
        H[len(self.Q) + len(self.Γ) + 1 :, :] = S @ self.P_h_symbol
        b[len(self.Q) + len(self.Γ) + 1, 0] = Rational("1/2")
        b = b + b_prime

        G = sp.zeros(
            len(self.Q) + len(self.Γ) + 1 + len(self.Q) * len(self.Γ),
            len(self.Q) * len(self.Γ),
            dtype=Rational,
        )
        G[len(self.Q) + len(self.Γ) + 1 :, :] = sp.eye(
            len(self.Q) + len(self.Γ), dtype=Rational
        )

        return H + G @ self.R_state @ self.P_h_state, b

    def construct_h_2_parameters(self) -> Tuple[sp.Matrix, sp.Matrix]:
        H = sp.zeros(
            len(self.Q) + len(self.Γ) + 1 + len(self.Q) * len(self.Γ) * 2,
            len(self.Q) + len(self.Γ) + 1 + len(self.Q) * len(self.Γ),
            dtype=Rational,
        )
        b = sp.zeros(
            len(self.Q) + len(self.Γ) + 1 + len(self.Q) * len(self.Γ) * 2,
            1,
            dtype=Rational,
        )
        H[: len(self.Q) + len(self.Γ), : len(self.Q) + len(self.Γ)] = sp.eye(
            len(self.Q) + len(self.Γ), dtype=Rational
        )
        H[len(self.Q) + len(self.Γ), len(self.Q) + len(self.Γ)] = Rational(2)
        H[len(self.Q) + len(self.Γ) + 1 :, :] = self.construct_transition_matrix()
        b[len(self.Q) + len(self.Γ) + 1, 0] = Rational(-1)

        return H, b

    def construct_h_3_parameters(self) -> Tuple[sp.Matrix, sp.Matrix]:
        H = sp.zeros(
            len(self.Q) + len(self.Γ) + 1 + len(self.Q) + len(self.Γ) + 1,
            len(self.Q) + len(self.Γ) + 1 + len(self.Q) * len(self.Γ) * 2,
            dtype=Rational,
        )
        b = sp.zeros(
            len(self.Q) + len(self.Γ) + 1 + len(self.Q) + len(self.Γ) + 1,
            1,
            dtype=Rational,
        )
        H[: len(self.Q) + len(self.Γ) + 1, : len(self.Q) + len(self.Γ) + 1] = sp.eye(
            len(self.Q) + len(self.Γ) + 1, dtype=Rational
        )
        H[len(self.Q) + len(self.Γ) + 1 :, :] = self.construct_separation_matrix()

        return H, b

    def construct_h_4_parameters(self) -> Tuple[sp.Matrix, sp.Matrix]:
        H = sp.zeros(
            self.D,
            len(self.Q) + len(self.Γ) + 1 + len(self.Q) + len(self.Γ) + 1,
            dtype=Rational,
        )
        b = sp.zeros(
            self.D,
            1,
            dtype=Rational,
        )
        H[: len(self.Q) + len(self.Γ) + 1, : len(self.Q) + len(self.Γ) + 1] = Rational(
            -1
        ) * sp.eye(len(self.Q) + len(self.Γ) + 1, dtype=Rational)
        H[
            len(self.Q) + len(self.Γ) + 1 : 2 * (len(self.Q) + len(self.Γ) + 1),
            len(self.Q) + len(self.Γ) + 1 : 2 * (len(self.Q) + len(self.Γ) + 1),
        ] = sp.eye(len(self.Q) + len(self.Γ) + 1, dtype=Rational)
        H[
            2 * (len(self.Q) + len(self.Γ) + 1) :, len(self.Q) + len(self.Γ) + 1
        ] = Rational(1)

        return H, b

    def setup_block_1(self) -> None:
        """Sets up the first decoder block."""
        # To implement the identity in the self-attention layer, the self-attention
        # query, key and value functions are None.
        self.decoder_block1.Q = sp.zeros(self.D, dtype=Rational)
        self.decoder_block1.K = sp.zeros(self.D, dtype=Rational)
        self.decoder_block1.V = sp.zeros(self.D, dtype=Rational)

        # Construct the affine transformations of the output layer of the first decoder.
        # It is composed of a single-layer MLP that has several linear transformations.
        H_1, b_1 = self.construct_h_1_parameters()
        H_2, b_2 = self.construct_h_2_parameters()
        H_3, b_3 = self.construct_h_3_parameters()
        H_4, b_4 = self.construct_h_4_parameters()

        def decoder_1_output_function(x: sp.Matrix) -> sp.Matrix:
            """Returns the output of the first decoder block."""

            x = H_1 @ x + b_1

            for d in range(self.D):
                x[d] = σ.subs(sym_x, x[d, 0])

            x = H_2 @ x + b_2
            x = H_3 @ x + b_3
            x = H_4 @ x + b_4

            return x

        self.decoder_block1.O = decoder_1_output_function

    def setup_block_2(self) -> None:
        self.decoder_block2.Q = sp.zeros(self.D, dtype=Rational)
        self.decoder_block2.K = sp.zeros(self.D, dtype=Rational)

        self.decoder_block2.V = sp.zeros(self.D, dtype=Rational)
        # Copy over the direction values to be summed.
        self.decoder_block2.V[
            self.embedding.offset(component=2) + len(self.Q) + len(self.Γ) + 2,
            self.embedding.offset(component=2) + len(self.Q) + len(self.Γ),
        ] = Rational(1)
        self.decoder_block2.V[
            self.embedding.offset(component=2) + len(self.Q) + len(self.Γ) + 3,
            self.embedding.offset(component=2) + len(self.Q) + len(self.Γ) + 1,
        ] = Rational(1)

        self.decoder_block2.O = None

    def setup_block_3(self) -> None:
        self.decoder_block3.Q = sp.zeros(self.D, dtype=Rational)
        self.decoder_block3.Q[
            self.embedding.offset(component=4) + 1,
            self.embedding.offset(component=2) + len(self.Q) + len(self.Γ) + 2,
        ] = Rational(1)
        self.decoder_block3.Q[
            self.embedding.offset(component=4) + 2,
            self.embedding.offset(component=4) + 2,
        ] = Rational(1)
        self.decoder_block3.Q[
            self.embedding.offset(component=4) + 3,
            self.embedding.offset(component=4) + 3,
        ] = Rational("1/3")

        self.decoder_block3.K = sp.zeros(self.D, dtype=Rational)
        self.decoder_block3.K[
            self.embedding.offset(component=4) + 1,
            self.embedding.offset(component=4) + 2,
        ] = Rational(1)
        self.decoder_block3.K[
            self.embedding.offset(component=4) + 1,
            self.embedding.offset(component=2) + len(self.Q) + len(self.Γ) + 3,
        ] = Rational(-1)
        self.decoder_block3.K[
            self.embedding.offset(component=4) + 3,
            self.embedding.offset(component=4) + 3,
        ] = Rational(1)

        self.decoder_block3.V = sp.zeros(self.D, dtype=Rational)
        self.decoder_block3.V[
            self.embedding.offset(component=3)
            + len(self.Γ)
            + 1 : self.embedding.offset(component=3)
            + len(self.Γ)
            + 1
            + len(self.Γ),
            self.embedding.offset(component=2)
            + len(self.Q) : self.embedding.offset(component=2)
            + len(self.Q)
            + len(self.Γ),
        ] = Rational(1)
        self.decoder_block3.V[
            self.embedding.offset(component=3) + len(self.Γ) + 1 + len(self.Γ),
            self.embedding.offset(component=4) + 1,
        ] = Rational(-1)
        self.decoder_block3.b_v[
            self.embedding.offset(component=3) + len(self.Γ) + 1 + len(self.Γ), 0
        ] = Rational(-1)

        self.decoder_block3.O = None

    def construct_F(self) -> None:
        self.F_H_1, self.F_b_1 = self.construct_f_1_parameters()
        (
            self.F_H_2_1,
            self.F_b_2_1,
            self.F_H_2_2,
            self.F_b_2_2,
            self.F_H_2_3,
            self.F_b_2_3,
            self.F_H_2_4,
            self.F_b_2_4,
        ) = self.construct_f_2_parameters()
        self.F_H_3, self.F_b_3 = self.construct_f_3_parameters()
        self.F_H_4, self.F_b_4 = self.construct_f_4_parameters()

    def construct_f_1_parameters(self) -> Tuple[sp.Matrix, sp.Matrix]:
        H = sp.zeros(
            len(self.Q) + 2 + len(self.Q) + 1 + len(self.Γ) + len(self.Γ) + 1,
            self.D,
            dtype=Rational,
        )
        b = sp.zeros(
            len(self.Q) + 2 + len(self.Q) + 1 + len(self.Γ) + len(self.Γ) + 1,
            1,
            dtype=Rational,
        )

        H[
            : len(self.Q),
            self.embedding.offset(component=1) : self.embedding.offset(component=1)
            + len(self.Q),
        ] = sp.eye(len(self.Q), dtype=Rational)
        H[
            len(self.Q) + 1,
            self.embedding.offset(component=2) + len(self.Q) + len(self.Γ),
        ] = Rational("1/2")
        H[
            len(self.Q) + 2,
            self.embedding.offset(component=2) + len(self.Q) + len(self.Γ),
        ] = Rational("-1/2")
        b[len(self.Q) + 1] = Rational("1/2")
        b[len(self.Q) + 2] = Rational("1/2")
        H[
            len(self.Q) + 2 : len(self.Q) + 2 + len(self.Q),
            self.embedding.offset(component=3) : self.embedding.offset(component=3)
            + len(self.Q),
        ] = Rational(1)
        H[
            len(self.Q) + 2 + len(self.Q), self.embedding.offset(component=4) + 1
        ] = Rational(1)
        H[
            len(self.Q) + 2 + len(self.Q), self.embedding.offset(component=3) + 1
        ] = Rational(-1)
        H[
            len(self.Q)
            + 2
            + len(self.Q)
            + 1 : len(self.Q)
            + 2
            + len(self.Q)
            + 1
            + len(self.Γ),
            self.embedding.offset(component=3)
            + len(self.Q)
            + 1 : self.embedding.offset(component=3)
            + 1
            + len(self.Γ),
        ] = sp.eye(len(self.Γ), dtype=Rational)
        b[:, 0] = one_hot(self.T.sym2idx[EMPTY], len(self.Γ))
        H[
            -1,
            self.embedding.offset(component=3) + len(self.Q) + 1 + len(self.Γ),
        ] = Rational(1)
        H[
            -1,
            self.embedding.offset(component=4) + 1,
        ] = Rational(-1)
        b[-1] = Rational(2)

        return H, b

    def construct_f_2_parameters(
        self,
    ) -> Tuple[
        sp.Matrix,
        sp.Matrix,
        sp.Matrix,
        sp.Matrix,
        sp.Matrix,
        sp.Matrix,
        sp.Matrix,
        sp.Matrix,
    ]:
        H_1 = sp.zeros(
            len(self.Q) + 2 + len(self.Q) + 1 + len(self.Γ) + len(self.Γ),
            len(self.Q) + 2 + len(self.Q) + 1 + len(self.Γ) + len(self.Γ) + 1,
            dtype=Rational,
        )
        b_1 = sp.zeros(
            len(self.Q) + 2 + len(self.Q) + 1 + len(self.Γ),
            1,
            dtype=Rational,
        )
        H_1[
            : len(self.Q) + 2 + len(self.Q) + 1, : len(self.Q) + 2 + len(self.Q) + 1
        ] = sp.eye(len(self.Q) + 2 + len(self.Q) + 1, dtype=Rational)
        H_1[
            len(self.Q)
            + 2
            + len(self.Q)
            + 1 : len(self.Q)
            + 2
            + len(self.Q)
            + 1
            + len(self.Γ),
            -1,
        ] = Rational(-1) * sp.ones(len(self.Γ), 1, dtype=Rational)
        H_1[
            len(self.Q)
            + 2
            + len(self.Q)
            + 1
            + len(self.Γ) : len(self.Q)
            + 2
            + len(self.Q)
            + 1
            + len(self.Γ)
            + len(self.Γ),
            -1,
        ] = sp.ones(len(self.Γ), 1, dtype=Rational)
        b_1[
            len(self.Q)
            + 2
            + len(self.Q)
            + 1
            + len(self.Γ) : len(self.Q)
            + 2
            + len(self.Q)
            + 1
            + len(self.Γ)
            + len(self.Γ),
            0,
        ] = Rational(-1) * sp.ones(len(self.Γ), 1, dtype=Rational)

        H_2 = sp.zeros(
            len(self.Q) + 2 + len(self.Q) + 1 + len(self.Γ),
            len(self.Q) + 2 + len(self.Q) + 1 + len(self.Γ) + len(self.Γ),
            dtype=Rational,
        )
        b_2 = sp.zeros(
            len(self.Q) + 2 + len(self.Q) + 1 + len(self.Γ),
            1,
            dtype=Rational,
        )
        H_2[
            : len(self.Q) + 2 + len(self.Q) + 1, : len(self.Q) + 2 + len(self.Q) + 1
        ] = sp.eye(len(self.Q) + 2 + len(self.Q) + 1, dtype=Rational)
        H_2[
            len(self.Q) + 2 + len(self.Q) + 1 :,
            len(self.Q)
            + 2
            + len(self.Q)
            + 1 : len(self.Q)
            + 2
            + len(self.Q)
            + 1
            + len(self.Γ),
        ] = sp.eye(len(self.Γ), dtype=Rational)
        H_2[
            len(self.Q) + 2 + len(self.Q) + 1 :,
            len(self.Q)
            + 2
            + len(self.Q)
            + 1
            + len(self.Γ) : len(self.Q)
            + 2
            + len(self.Q)
            + 1
            + len(self.Γ)
            + len(self.Γ),
        ] = sp.eye(len(self.Γ), dtype=Rational)

        H_3 = sp.zeros(
            len(self.Q) + 2 + len(self.Q) + len(self.Γ),
            len(self.Q) + 2 + len(self.Q) + 1 + len(self.Γ),
            dtype=Rational,
        )
        H_3[
            len(self.Q) + 2 : len(self.Q) + 2 + len(self.Q),
            len(self.Q) + 2 + len(self.Q),
        ] = Rational(-1) * sp.ones(len(self.Γ), 1, dtype=Rational)
        H_3[
            len(self.Q) + 2 + len(self.Q) :,
            len(self.Q) + 2 + len(self.Q),
        ] = sp.ones(len(self.Γ), 1, dtype=Rational)

        b_3 = sp.zeros(
            len(self.Q) + 2 + len(self.Q) + len(self.Γ),
            1,
            dtype=Rational,
        )
        b_3[len(self.Q) + 2 + len(self.Q) :, 0] = Rational(-1) * sp.ones(
            len(self.Γ), 1, dtype=Rational
        )

        H_4 = sp.zeros(
            len(self.Q) + 2 + len(self.Q),
            len(self.Q) + 2 + len(self.Q) + len(self.Γ),
            dtype=Rational,
        )
        H_4[
            len(self.Q) + 2 : len(self.Q) + 2 + len(self.Q),
            len(self.Q) + 2 : len(self.Q) + 2 + len(self.Q),
        ] = sp.eye(len(self.Γ), dtype=Rational)
        H_4[
            len(self.Q) + 2 + len(self.Q) :,
            len(self.Q) + 2 : len(self.Q) + 2 + len(self.Q),
        ] = sp.eye(len(self.Γ), dtype=Rational)

        b_4 = sp.zeros(
            len(self.Q) + 2 + len(self.Q),
            1,
            dtype=Rational,
        )
        return H_1, b_1, H_2, b_2, H_3, b_3, H_4, b_4

    def construct_f_3_parameters(
        self,
    ) -> Tuple[sp.Matrix, sp.Matrix]:
        H = sp.zeros(
            len(self.Q) + 2 + len(self.Q) + 1 + len(self.Γ),
            len(self.Q) + 2 + len(self.Q) + 1 + len(self.Γ) + len(self.Γ) + 1,
            dtype=Rational,
        )
        b = sp.zeros(
            len(self.Q) + 2 + len(self.Q) + 1 + len(self.Γ),
            1,
            dtype=Rational,
        )

        H[: len(self.Q), : len(self.Q)] = sp.eye(len(self.Q), dtype=Rational)
        H[len(self.Q) : len(self.Q) + len(self.Γ), len(self.Q) + 1 :] = sp.eye(
            len(self.Q), dtype=Rational
        )
        H[len(self.Q) + len(self.Γ), len(self.Q)] = Rational(1)
        H[len(self.Q) + len(self.Γ), len(self.Q) + 1] = Rational(-1)
        return H, b

    def construct_f_4_parameters(
        self,
    ) -> Tuple[sp.Matrix, sp.Matrix]:
        H = sp.zeros(
            self.D,
            len(self.Q) + 2 + len(self.Γ),
            dtype=Rational,
        )
        b = sp.zeros(
            self.D,
            1,
            dtype=Rational,
        )
        return H, b

    def F(self, x: sp.Matrix) -> sp.Matrix:
        x = self.F_H_1 @ x + self.F_b_1

        for d in range(self.D):
            x[d] = σ.subs(sym_x, x[d, 0])

        x = self.F_H_2_1 @ x + self.F_b_2_1

        for d in range(self.D):
            x[d] = σ.subs(sym_x, x[d, 0])

        x = self.F_H_2_2 @ x + self.F_b_2_2

        x = self.F_H_2_3 @ x + self.F_b_2_3

        for d in range(self.D):
            x[d] = σ.subs(sym_x, x[d, 0])

        x = self.F_H_2_4 @ x + self.F_b_2_4

        x = self.F_H_3 @ x + self.F_b_3
        x = self.F_H_4 @ x + self.F_b_4

        return x

    def __call__(self, X: sp.Matrix, K_e: sp.Matrix, V_e: sp.Matrix) -> sp.Matrix:
        Z_1 = self.decoder_block1(X, K_e, V_e)
        Z_2 = self.decoder_block2(Z_1, K_e, V_e)
        Z_3 = self.decoder_block2(Z_2, K_e, V_e)

        y_r_1 = self.F(sp.Matrix(Z_3[-1, :], dtype=Rational))

        return y_r_1


def α(y: String, n: int) -> Sym:
    if n < len(y):
        return y[n]
    else:
        return y[-1]


def β(y: String, n: int) -> int:
    if n < len(y):
        return n
    else:
        return len(y) - 1


def get_initial_state(embedding: Embedding) -> sp.Matrix:
    h_0 = sp.zeros(embedding.D, 1, dtype=Rational)

    h_0[
        embedding.offset(component=0) + embedding.T.q2idx[embedding.T.q_init], 0
    ] = Rational(1)
    h_0[
        embedding.offset(component=0) + len(embedding.T.Γ) + embedding.T.sym2idx[EMPTY],
        0,
    ] = Rational(1)

    h_0 = h_0 + embedding.position_encoding(0)

    return h_0

from typing import Tuple
from itertools import product

from sympy import Rational
import sympy as sp

from turnformer.base.symbol import EMPTY, Sym
from turnformer.base.string import String
from turnformer.base.turing_machine import Direction
from turnformer.transformer.symbol_encoding import Embedding
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

    def f_1(self, x: sp.Matrix) -> sp.Matrix:
        """Produces the one-hot endocing of the current state and symbol
        based on the current hidden state.
        """
        v = sp.zeros(len(self.Γ) * len(self.Q), 1, dtype=Rational)

        # Take the embeddings of the current state and symbol.
        e_q = x[: len(self.Q)]
        e_s = x[len(self.Q) : len(self.Q) + len(self.Γ)]
        # S_n @ e_s is a vector of ones iff the current symbol is the n-th symbol in
        # the alphabet.

        for n in range(len(self.Γ)):
            v[n * len(self.Q) : (n + 1) * len(self.Q)] = e_q + self.S_n(n) @ e_s

        v = v - sp.ones(len(self.Γ) * len(self.Q), 1, dtype=Rational)
        # This will be 1 exactly at the position corresponding to the pair (q, s),
        # where q is the current state and s is the current symbol.

        return v

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

    def construct_A_matrix(self) -> sp.Matrix:
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

    def construct_B_matrix(self) -> Tuple[sp.Matrix, sp.Matrix]:
        B = sp.zeros(
            len(self.Q) + len(self.Γ) + 1 + len(self.Q) * len(self.Γ),
            self.D,
            dtype=Rational,
        )
        b = sp.zeros(
            len(self.Q) + len(self.Γ) + 1 + len(self.Q) * len(self.Γ), 1, dtype=Rational
        )

        B[: len(self.Q) + len(self.Γ), : len(self.Q) + len(self.Γ)] = sp.eye(
            len(self.Q) + len(self.Γ), dtype=Rational
        )
        B[len(self.Q) + len(self.Γ), len(self.Q) + len(self.Γ)] = Rational("1/2")
        S, b_prime = self.construct_S_matrix()
        B[len(self.Q) + len(self.Γ) + 1 :, :] = S @ self.P_h_symbol
        b[len(self.Q) + len(self.Γ) + 1, 0] = Rational("1/2")
        b = b + b_prime

        return B, b

    def setup_block_1(self) -> None:
        """Sets up the first decoder block."""
        # To implement the identity in the self-attention layer, the self-attention
        # key and value matrices are zero matrices.
        self.decoder_block1.K = sp.zeros(self.decoder_block1.D, dtype=Rational)
        self.decoder_block1.V = sp.zeros(self.decoder_block1.D, dtype=Rational)

        T = self.construct_transition_matrix()

        A = self.construct_A_matrix()

        B, b = self.construct_B_matrix()

        def decoder_1_output_function(x: sp.Matrix) -> sp.Matrix:
            """Returns the output of the first decoder block."""

            e_q = self.P_h_state @ x

            r_q = self.R_state @ e_q
            b_q = sp.zeros(
                len(self.Q) + len(self.Γ) + 1 + len(self.Q) * len(self.Γ),
                1,
                dtype=Rational,
            )
            b_q[len(self.Q) + len(self.Γ) + 1 :, 0] = r_q

            for d in range(self.D):
                x[d] = σ.subs(sym_x, (B @ x + b_q + b)[d, 0])

        self.decoder_block1.O = decoder_1_output_function

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

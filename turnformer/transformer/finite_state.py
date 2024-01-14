from itertools import product
from typing import Tuple, Union

import numpy as np

from rayuela.base.semiring import Real
from rayuela.base.symbol import EOS, Sym
from rayuela.fsa.fsa import FSA
from rayuela.fsa.state import State
from rayuela.nn.F import H
from rayuela.nn.modules import construct_and
from rayuela.nn.transformer.transformer import Transformer, TransformerLayer
from rayuela.nn.transformer.utils import ProjectionFunctions


class FiniteStateTransform:
    def __init__(self, A: FSA, projection: str = "softmax") -> None:
        """The class performing the Minsky transformation of an FSA into an Heaviside
        Elman RNN.

        Args:
            A (FSA): The FSA to be transformed.
            projection (str): What kind of projection from the scores to the normalized
                local probability distribution to use.
                Can be either "softmax" or "sparsemax", where "sparsemax" can only be
                used if the FSA is probabilistic.
                Defaults to "softmax".
        """
        assert A.deterministic, "The FSA must be deterministic."  # TODO
        assert A.R == Real

        self.A = A
        self.q0 = list(self.A.I)[0][0]
        self.Sigma = list(self.A.Sigma)
        self.SigmaEOS = self.Sigma + [EOS]
        self.n_states, self.n_symbols = len(self.A.Q), len(self.Sigma)
        self.D1 = self.n_symbols + 1
        self.D2 = 2
        self.D3 = self.n_states
        self.D = self.D1 + self.D2 + self.D3  # TBD

        self.construct()

    def set_up_orderings(self):
        # Ordering of Σ x Q
        self.n = dict()
        self.n_inv = dict()
        for i, (q, a) in enumerate(product(self.A.Q, self.SigmaEOS)):
            self.n[(q, a)] = i
            self.n_inv[i] = (q, a)

        # Ordering of Σbar
        self.m = {a: i for i, a in enumerate(self.SigmaEOS)}
        self.m_inv = {i: a for i, a in enumerate(self.SigmaEOS)}

        # Ordering of Q
        self.s = {q: i for i, q in enumerate(self.A.Q)}
        self.s_inv = {i: q for i, q in enumerate(self.A.Q)}

    def one_hot(self, x: Union[State, str, Sym, Tuple[State, Sym]]) -> np.ndarray:
        if isinstance(x, str):
            x = Sym(x)

        if isinstance(x, Sym):
            y = np.zeros((self.n_symbols + 1))
            y[self.m[x]] = 1
            return y
        elif isinstance(x, State):
            y = np.zeros((self.n_states))
            y[self.s[x]] = 1
            return y
        elif isinstance(x, tuple):
            y = np.zeros((self.n_states * (self.n_symbols + 1)))
            y[self.n[x]] = 1
            return y
        else:
            raise TypeError
        return y

    def e2qy(self, e: np.ndarray) -> Tuple[State, Sym]:
        return self.n_inv[np.argmax(e)]

    def eq2q(self, e: np.ndarray) -> Tuple[State, Sym]:
        return self.s_inv[np.argmax(e)]

    def ey2y(self, e: np.ndarray) -> Tuple[State, Sym]:
        return self.m_inv[np.argmax(e)]

    def initial_static_encoding(self) -> np.ndarray:
        # _state = self.one_hot(self.q0)
        # _symbol = self.one_hot(self.Sigma[0])
        # return np.concatenate((_state, _symbol))
        return self.one_hot(self.q0)

    def positional_encoding(self, t: int) -> np.ndarray:
        return np.asarray([1, t])

    def construct_and_gates(self) -> Tuple[np.ndarray, np.ndarray]:
        W = np.zeros((self.n_states * (self.n_symbols + 1), self.D))
        b = np.zeros((self.n_states * (self.n_symbols + 1)))
        for q, y in product(self.A.Q, self.SigmaEOS):
            _w, _b = construct_and(
                self.D, [self.m[y], (self.n_symbols + 1) + 2 + self.s[q]]
            )
            W[self.n[(q, y)], :] = _w
            b[self.n[(q, y)]] = _b

        return W, b

    def setup_layer_1(self):
        """Construct the parameters of the first transformer block.
        This layer is responsible for the computation of the previous state and
        current input symbol one-hot encoding.
        """

        # At this point, the symbol representation will be of the form
        # [yt, pt-1, qt-1]
        # The query and the key matrices should project out the positional encoding
        # while the value matrix should project out the state and symbol representations
        Wq = np.zeros((2, self.D))
        Wq[:, (self.n_symbols + 1) : (self.n_symbols + 1) + 2] = np.eye(2)
        bq = np.asarray([0, -1])

        Wk = np.zeros((2, self.D))
        P = np.zeros((2, 2))
        P[0, 1] = -1
        P[1, 0] = 1
        Wk[:, (self.n_symbols + 1) : (self.n_symbols + 1) + 2] = P

        # Wv = np.zeros((self.D, self.D))
        # Wv[: (self.n_symbols + 1), : (self.n_symbols + 1)] = np.eye(
        #     (self.n_symbols + 1)
        # )
        # Wv[-self.n_states :, -self.n_states :] = np.eye(self.n_states)
        Wv = np.eye(self.D)

        def Q(X):
            return (Wq @ X.T).T + bq

        def K(X):
            return (Wk @ X.T).T

        def V(X):
            return (Wv @ X.T).T

        def f(q, k):
            return -np.abs(np.dot(q, k.T))

        Wo, bo = self.construct_and_gates()
        # Woʼ = np.zeros((2, self.D))
        # Woʼ[:, (self.n_symbols + 1) : (self.n_symbols + 1) + 2] = np.eye(2)
        # boʼ = np.zeros((2))
        # Wo = np.vstack([Wo, Woʼ])
        # bo = np.hstack([bo, boʼ])
        self.Wo = Wo
        self.bo = bo

        def O(X):  # noqa: E741, E743
            return H((Wo @ X.T).T + bo)

        self.T1 = TransformerLayer(
            Q=Q,
            K=K,
            V=V,
            f=f,
            projection=ProjectionFunctions.unique_hard_normalization,
            O=O,
        )

    def setup_layer_2(self):
        """Construct the parameters of the second transformer block."""

        # At this point, the symbol representation will be of the form
        # [yt, pt-1, qt-1]
        # The query and the key matrices should project out the positional encoding
        # while the value matrix should project out the state and symbol representations
        Wq = np.zeros((2, self.n_states * (self.n_symbols + 1) + 2))
        Wq[:, -2:] = np.eye(2)

        Wk = np.zeros((2, self.n_states * (self.n_symbols + 1) + 2))
        P = np.zeros((2, 2))
        P[0, 1] = 1
        P[1, 0] = -1
        Wk[:, -2:] = P

        Wv = np.zeros((self.n_states, self.n_states * (self.n_symbols + 1) + 2))
        for q in self.A.Q:
            for a, qʼ, _ in self.A.arcs(q):
                Wv[self.s[qʼ], self.n[(q, a)]] = 1

        def Q(X):
            return (Wq @ X.T).T

        def K(X):
            return (Wk @ X.T).T

        def V(X):
            return (Wv @ X.T).T

        def f(q, k):
            return -np.abs(np.dot(q, k.T))

        def O(X):  # noqa: E741, E743
            return X

        self.T2 = TransformerLayer(
            Q=Q,
            K=K,
            V=V,
            f=f,
            projection=ProjectionFunctions.unique_hard_normalization,
            O=O,
        )

    def setup_output_matrix(self):
        self.E = -np.inf * np.ones((self.n_symbols + 1, self.n_states))

        for q in self.A.Q:
            for a, _, w in self.A.arcs(q):
                self.E[self.m[a], self.s[q]] = np.log(w.value)

        for q, w in self.A.F:
            # The final weight is an alternative "output" weight
            # for the final states.
            self.E[self.m[EOS], self.s[q]] = np.log(w.value)

    def construct(self):
        self.set_up_orderings()

        # Set up layer 1:
        self.setup_layer_1()

        # Set up layer 2:
        self.setup_layer_2()

        # Set up the output matrix
        self.setup_output_matrix()

        def F(x):
            return x

        self.T = Transformer(
            layers=[self.T1, self.T2],
            # layers=[self.T1],
            F=F,
            encoding=self.one_hot,
            positional_encoding=self.positional_encoding,
            S=self.initial_static_encoding(),
            Tf=self,
        )

from itertools import product
from typing import Tuple, Union

import numpy as np
from rayuela.fsa.fsa import FSA
from rayuela.fsa.state import State

from turnformer.base.F import H, ProjectionFunctions, ReLU
from turnformer.base.modules import construct_and
from turnformer.base.symbols import EOS
from turnformer.transformer.transformer import (
    AttentionHead,
    MultiHeadAttentionLayer,
    TransfomerLM,
    Transformer,
)


class FiniteStateTransform:
    def __init__(self, A: FSA, projection: str = "softmax") -> None:
        """The class that constructs a transformer network simulating a probabilistic
        finite state automaton.

        Args:
            A (FSA): The PFSA to be transformed.
            projection (str): What kind of projection from the scores to the normalized
                local probability distribution to use.
                Can be either "softmax" or "sparsemax", where "sparsemax" can only be
                used if the FSA is probabilistic.
                Defaults to "softmax".
        """
        assert A.deterministic, "The FSA must be deterministic."  # TODO
        assert A.probabilistic, "The FSA must be probabilistic."

        self.A = A
        self.q0 = list(self.A.I)[0][0]
        self.Sigma = [str(a) for a in self.A.Sigma]
        self.SigmaEOS = self.Sigma + [EOS]
        self.n_states, self.n_symbols = len(self.A.Q), len(self.Sigma)

        self.D1 = self.n_symbols + 1
        self.D2 = 2
        self.D3 = self.n_states
        self.D4 = self.n_states * (self.n_symbols + 1)

        self.C1 = self.D1
        self.C2 = self.D1 + self.D2
        self.C3 = self.D1 + self.D2 + self.D3
        self.C4 = self.D1 + self.D2 + self.D3 + self.D4

        self.D = self.D1 + self.D2 + self.D3 + self.D4

        # The hidden states have the organization:
        # [
        #   one-hot(yt)                 # |Sigma| + 1
        #   positional encoding,        # 2
        #   one-hot(qt-1)               # |Q|
        #   one-hot(qt-1, yt)           # |Q| * (|Sigma| + 1)  [processing, temporary]
        # ]

        self.construct()

    def display_hidden_state(self, X: np.ndarray) -> None:
        print()
        for i, x in enumerate(X):
            for j in range(self.n_symbols + 1):
                if x[j] == 1:
                    print(f"y_{i}: {self.SigmaEOS[j]}")
                    break
            print(f"p_{i}: {int(x[self.C1 + 1])}")
            if x[self.C2 : self.C3].sum() > 0:
                print(f"q_{i}: {self.s_inv[np.argmax(x[self.C2: self.C3])]}")
            if x[self.C3 : self.C4].sum() > 0:
                print(f"q'_{i},y': {self.n_inv[np.argmax(x[self.C3: self.C4])]}")
            print("---------")

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

    def one_hot(self, x: Union[State, str, Tuple[State, str]]) -> np.ndarray:
        if isinstance(x, str):
            y = np.zeros((self.n_symbols + 1))
            if x != "":
                # For the special cases of the beginning of the string or empty string
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

    def e2qy(self, x: np.ndarray) -> Tuple[State, str]:
        return self.n_inv[np.argmax(x[self.C3 : self.C4])]

    def eq2q(self, x: np.ndarray) -> State:
        return self.s_inv[np.argmax(x[self.C2 : self.C3])]

    def ey2y(self, x: np.ndarray) -> str:
        return self.m_inv[np.argmax(x[: self.C1])]

    def initial_static_encoding(self, y: str, t: int) -> np.ndarray:
        X0 = np.concatenate(
            [
                self.one_hot(y),
                self.positional_encoding(t),
                self.one_hot(self.q0) if t == 0 else np.zeros(self.n_states),
                np.zeros(self.n_states * (self.n_symbols + 1)),
            ]
        )

        return X0

    def positional_encoding(self, t: int) -> np.ndarray:
        return np.asarray([1, t + 1])

    def construct_transition_gates(self) -> Tuple[np.ndarray, np.ndarray]:
        W = np.zeros((self.D, self.D))
        b = np.zeros((self.D))
        for q in self.A.Q:
            for y, qʼ, _ in self.A.arcs(q):
                _w, _b = construct_and(self.D, [self.m[str(y)], self.C2 + self.s[q]])
                W[self.C3 + self.n[(qʼ, str(y))], :] = np.maximum(
                    W[self.C3 + self.n[(qʼ, str(y))], :], _w.reshape((-1,))
                )
                b[self.C3 + self.n[(qʼ, str(y))]] = np.minimum(
                    _b, b[self.C3 + self.n[(qʼ, str(y))]]
                )

        return W, b

    def construct_copy_head(self) -> AttentionHead:
        """Construct a transformer head that simply copies the content of the tape.

        Returns:
            AttentionHead: The transformer head.
        """
        # At this point, the symbol representation will be of the form
        # [yt, pt-1, qt-1]
        # The query and the key matrices should project out the positional encoding
        # while the value matrix should project out the state and symbol representations
        Wq = np.zeros((2, self.D))
        Wq[:, self.C1 : self.C2] = np.eye(2)
        bq = np.asarray([0, 0])

        Wk = np.zeros((2, self.D))
        P = np.zeros((2, 2))
        P[0, 1] = -1
        P[1, 0] = 1
        Wk[:, self.C1 : self.C2] = P

        Wv = np.eye(self.D)

        def Q(X):
            return (Wq @ X.T).T + bq

        def K(X):
            return (Wk @ X.T).T

        def V(X):
            return (Wv @ X.T).T

        def f(q, k):
            return -np.abs(np.dot(q, k.T))

        def O(X):  # noqa: E741, E743
            return X

        return AttentionHead(
            Q=Q,
            K=K,
            V=V,
            f=f,
            projection=ProjectionFunctions.unique_hard,
            O=O,
        )

    def construct_transition_head(self) -> AttentionHead:
        """Constructs the transformer head that computes the next state of the FSA
        given the current state and the input symbol.

        Returns:
            AttentionHead: The transformer head.
        """
        Wq = np.zeros((2, self.D))
        Wq[:, self.C1 : self.C2] = np.eye(2)
        bq = np.asarray([0, -1])

        Wk = np.zeros((2, self.D))
        P = np.zeros((2, 2))
        P[0, 1] = -1
        P[1, 0] = 1
        Wk[:, self.C1 : self.C2] = P

        Wv = np.eye(self.D)

        def Q(X):
            return (Wq @ X.T).T + bq

        def K(X):
            return (Wk @ X.T).T

        def V(X):
            return (Wv @ X.T).T

        def f(q, k):
            return -np.abs(np.dot(q, k.T))

        Wo, bo = self.construct_transition_gates()

        def O(X):  # noqa: E741, E743
            return ReLU((Wo @ X.T).T + bo)

        return AttentionHead(
            Q=Q,
            K=K,
            V=V,
            f=f,
            projection=ProjectionFunctions.unique_hard,
            O=O,
        )

    def construct_layer(self):
        """Construct the parameters of the first transformer block.
        This layer is responsible for the computation of the previous state and
        current input symbol one-hot encoding.
        """

        H1 = self.construct_copy_head()
        H2 = self.construct_transition_head()

        P1 = np.zeros((self.D3, 2 * self.D))
        P1[:, self.C2 : self.C3] = np.eye(self.D3)
        P2 = np.zeros((self.D4, 2 * self.D))
        P2[:, -self.D4 :] = np.eye(self.D4)
        O2 = np.zeros((self.n_states, self.D4))
        for q, y in product(self.A.Q, self.SigmaEOS):
            O2[self.s[q], self.n[(q, y)]] = 1

        E = np.ones((self.D3, self.D3))

        W1 = np.zeros((self.D, self.D3))
        W1[self.C2 : self.C3, :] = np.eye(self.D3)

        W2 = np.zeros((self.D, 2 * self.D))
        W2[: self.C2, : self.C2] = np.eye(self.C2)

        def fH(Z):
            v1 = (P1 @ Z.T).T
            v2 = (P2 @ Z.T).T
            v2 = H(O2 @ v2.T).T  # This can be replaced by a composition of ReLUs
            Zʹ = v1 + H(v2 - (E @ v1.T).T)

            return (W2 @ Z.T + W1 @ Zʹ.T).T

        return MultiHeadAttentionLayer(heads=[H1, H2], fH=fH)

    def construct_output_matrix(self):
        E = -np.inf * np.ones((self.n_symbols + 1, self.n_states))

        for q in self.A.Q:
            for a, _, w in self.A.arcs(q):
                E[self.m[str(a)], self.s[q]] = np.log(w.value)

        for q, w in self.A.F:
            # The final weight is an alternative "output" weight
            # for the final states.
            E[self.m[EOS], self.s[q]] = np.log(w.value)

        return E

    def construct(self):
        self.set_up_orderings()

        # Set up layer:
        MAH = self.construct_layer()

        # Set up the output matrix
        E = self.construct_output_matrix()

        Wf = np.zeros((self.n_states, self.D))
        Wf[:, self.C2 : self.C3] = np.eye(self.n_states)

        def F(x):
            return (Wf @ x.T).T

        T = Transformer(
            layers=[MAH],
            F=F,
            encoding=self.one_hot,
            positional_encoding=self.positional_encoding,
            X0=self.initial_static_encoding,
            Tf=self,
        )

        self.lm = TransfomerLM(T, E)

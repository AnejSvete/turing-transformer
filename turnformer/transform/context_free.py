from itertools import product
from typing import Tuple, Union

import numpy as np

from turnformer.automata.pda import SingleStackPDA
from turnformer.base.F import ProjectionFunctions, ReLU
from turnformer.base.modules import construct_and
from turnformer.base.symbols import BOT, EOS
from turnformer.transformer.transformer import (
    AttentionHead,
    MultiHeadAttentionLayer,
    Transformer,
)


class ContextFreeTransform:
    def __init__(self, P: SingleStackPDA, projection: str = "softmax") -> None:
        """The class that constructs a transformer network simulating a probabilistic
        single-stack pushdown automaton.

        Args:
            A (FSA): The PFSA to be transformed.
            projection (str): What kind of projection from the scores to the normalized
                local probability distribution to use.
                Can be either "softmax" or "sparsemax", where "sparsemax" can only be
                used if the FSA is probabilistic.
                Defaults to "softmax".
        """
        assert P.probabilistic, "The FSA must be probabilistic."

        self.P = P
        self.q0 = 0
        self.Sigma = P.Σ
        self.SigmaEOS = self.Sigma + [EOS]
        self.Gamma = P.Γ
        self.n_states, self.n_symbols = len(self.P.Q), len(self.Sigma)
        self.n_stack_symbols = len(self.Gamma)

        self.D1 = self.n_symbols + 1  # One-hot encoding of the input symbol
        self.D2 = 3  # Positional encoding
        self.D3 = self.n_states  # One-hot encoding of the current state
        self.D4 = self.n_stack_symbols  # One-hot encoding of the stack symbol
        self.D5 = 1  # The action
        # TEMPORARY, PROCESSING:
        self.D6 = self.n_stack_symbols  # One-hot encoding of the next stack symbol
        self.D7 = 3  # One-hot encoding of the next action
        self.D8 = 1  # c
        # One-hot encoding of the next state and current symbol pairs
        self.D9 = self.n_states * (self.n_symbols + 1)

        self.C1 = self.D1
        self.C2 = self.C1 + self.D2
        self.C3 = self.C2 + self.D3
        self.C4 = self.C3 + self.D4
        self.C5 = self.C4 + self.D5
        self.C6 = self.C5 + self.D6
        self.C7 = self.C6 + self.D7
        self.C8 = self.C7 + self.D8
        self.C9 = self.C8 + self.D9

        self.D = self.C9

        # The hidden states have the organization:
        # [
        #   one-hot(yt)                 # |Sigma| + 1
        #   positional encoding,        # 2
        #   one-hot(qt-1)               # |Q|
        #   one-hot(γt-1)               # |Γ|
        #   at-1                        # 1
        # TEMPORARY, PROCESSING:
        #   one-hot(γt)                 # |Γ|
        #   one-hot(at-1)               # 3
        #   ct-1                        # 1
        #   one-hot(qt-1, yt)           # |Q| * (|Sigma| + 1)
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
        for i, (q, a) in enumerate(product(self.P.Q, self.SigmaEOS)):
            self.n[(q, a)] = i
            self.n_inv[i] = (q, a)

        # Ordering of Σbar
        self.m = {a: i for i, a in enumerate(self.SigmaEOS)}
        self.m_inv = {i: a for i, a in enumerate(self.SigmaEOS)}

        # Ordering of Q
        self.s = {q: i for i, q in enumerate(self.P.Q)}
        self.s_inv = {i: q for i, q in enumerate(self.P.Q)}

        # Ordering of Γ
        self.g = {γ: i for i, γ in enumerate(self.Gamma)}
        self.g_inv = {i: γ for i, γ in enumerate(self.Gamma)}

    def one_hot(
        self, x: Union[int, str, Tuple[int, str]], component: str
    ) -> np.ndarray:
        if component == "symbol":
            y = np.zeros((self.n_symbols + 1))
            if x != "":
                # For the special cases of the beginning of the string or empty string
                y[self.m[x]] = 1
            return y
        elif component == "state":
            y = np.zeros((self.n_states))
            y[self.s[x]] = 1
            return y
        elif component == "state-symbol":
            y = np.zeros((self.n_states * (self.n_symbols + 1)))
            y[self.n[x]] = 1
            return y
        elif component == "stack":
            y = np.zeros((self.n_stack_symbols))
            y[self.g[x]] = 1
            return y
        else:
            raise TypeError
        return y

    def e2qy(self, x: np.ndarray) -> Tuple[int, str]:
        return self.n_inv[np.argmax(x[-self.D6 :])]

    def eq2q(self, x: np.ndarray) -> int:
        return self.s_inv[np.argmax(x[self.C2 : self.C3])]

    def ey2y(self, x: np.ndarray) -> str:
        return self.m_inv[np.argmax(x[: self.C1])]

    def initial_static_encoding(self, y: str, t: int) -> np.ndarray:
        X0 = np.concatenate(
            [
                self.one_hot(y, "symbol"),
                self.positional_encoding(t),
                self.one_hot(self.q0, "state") if t == 0 else np.zeros(self.n_states),
                self.one_hot(BOT, "stack"),
                np.asarray([1]),
                np.zeros(self.n_stack_symbols),
                np.zeros(self.D7),
                np.zeros(self.D8),
                np.zeros(self.n_states * (self.n_symbols + 1)),
            ]
        )

        return X0

    def positional_encoding(self, t: int) -> np.ndarray:
        return np.asarray([1, t + 1, 1 / (t + 1)])

    def action_encoding(self, a: str) -> np.ndarray:
        if a == "PUSH":
            return np.asarray([1])
        elif a == "POP":
            return np.asarray([-1])
        elif a == "NOOP":
            return np.asarray([0])
        else:
            raise ValueError

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
        Wq = np.zeros((2, self.D))
        Wq[:, self.C1 : self.C1 + 2] = np.eye(2)
        bq = np.asarray([0, 0])

        Wk = np.zeros((2, self.D))
        P = np.zeros((2, 2))
        P[0, 1] = -1
        P[1, 0] = 1
        Wk[:, self.C1 : self.C1 + 2] = P

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

    def construct_stack_top_head(self) -> AttentionHead:
        """Construct a transformer head that computes the position
        of the top stack symbol.

        Returns:
            AttentionHead: The transformer head.
        """
        Wq = np.zeros((1, self.D))
        Wk = np.zeros((1, self.D))
        Wv = np.zeros((self.D, self.D))
        Wv[-self.D9 - 1, self.C4] = 1

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

        return AttentionHead(
            Q=Q,
            K=K,
            V=V,
            f=f,
            projection=ProjectionFunctions.averaging_hard,
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
        H2 = self.construct_stack_top_head()
        # H2 = self.construct_transition_head()

        # P1 = np.zeros((self.D3, 2 * self.D))
        # P1[:, self.C2 : self.C3] = np.eye(self.D3)
        # P2 = np.zeros((self.D4, 2 * self.D))
        # P2[:, -self.D4 :] = np.eye(self.D4)
        # O2 = np.zeros((self.n_states, self.D4))
        # for q, y in product(self.P.Q, self.SigmaEOS):
        #     O2[self.s[q], self.n[(q, y)]] = 1

        # E = np.ones((self.D3, self.D3))

        # W1 = np.zeros((self.D, self.D3))
        # W1[self.C2 : self.C3, :] = np.eye(self.D3)

        # W2 = np.zeros((self.D, 2 * self.D))
        # W2[: self.C2, : self.C2] = np.eye(self.C2)

        def fH(Z):
            # v1 = (P1 @ Z.T).T
            # v2 = (P2 @ Z.T).T
            # v2 = H(O2 @ v2.T).T  # This can be replaced by a composition of ReLUs
            # Zʹ = v1 + H(v2 - (E @ v1.T).T)

            # return (W2 @ Z.T + W1 @ Zʹ.T).T

            return Z

        return MultiHeadAttentionLayer(heads=[H1, H2], fH=fH)

    def construct_output_matrix(self):
        E = -np.inf * np.ones((self.n_symbols + 1, self.n_states))

        for q in self.P.Q:
            for a, _, w in self.P.arcs(q):
                E[self.m[str(a)], self.s[q]] = np.log(w.value)

        for q, w in self.P.F:
            # The final weight is an alternative "output" weight
            # for the final states.
            E[self.m[EOS], self.s[q]] = np.log(w.value)

        return E

    def construct(self):
        self.set_up_orderings()

        # Set up layer:
        MAH = self.construct_layer()

        # Set up the output matrix
        # E = self.construct_output_matrix()

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

        self.T = T

        # self.lm = TransfomerLM(T, E)

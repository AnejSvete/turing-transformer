from itertools import product
from typing import Tuple, Union

import numpy as np

from turnformer.automata.pda import Action, SingleStackPDA
from turnformer.base.F import H, ProjectionFunctions, ReLU
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
        self.n_actions = len(self.P.actions)
        self.n_states, self.n_symbols = len(self.P.Q), len(self.Sigma)
        self.n_stack_symbols = len(self.Gamma)

        # The hidden states have the organization:
        # [
        #   INPUT
        #       one-hot(y)                      # |Σ| + 1, 0 : C1
        #       positional encoding,            # 4, C1 : C2
        #   --------------------------------
        #   CURRENT CONFIGURATION
        #       one-hot(q↑)                     # |Q|, C2 : C3
        #       c↑                              # 1, C3 : C4
        #       l↑                              # 1, C4 : C5
        #       one-hot(γ↑)                     # |Γ|, C5 : C6
        #   --------------------------------
        #   TRANSITION
        #       one-hot(q↓)                     # |Q|, C6 : C7
        #       one-hot(γ↓)                     # |Γ|, C7 : C8
        #       one-hot(a↓)                     # |A|, C8 : C9
        #   --------------------------------
        #   TEMPORARY
        #       one-hot(q↓, a↓, γ↓, y)          # |Q| * |A| * |Γ| * (|Σ| + 1), C9 : C10
        # ]

        # -------------------------
        self.D1 = self.n_symbols + 1
        self.D2 = 4
        # ------------------
        self.D3 = self.n_states
        self.D4 = 1
        self.D5 = 1
        self.D6 = self.n_stack_symbols
        # ------------------
        self.D7 = self.n_states
        self.D8 = self.n_stack_symbols
        self.D9 = self.n_actions
        self.D10 = (
            self.n_states * self.n_actions * self.n_stack_symbols * (self.n_symbols + 1)
        )
        # -------------------------

        self.C1 = self.D1
        self.C2 = self.C1 + self.D2
        self.C3 = self.C2 + self.D3
        self.C4 = self.C3 + self.D4
        self.C5 = self.C4 + self.D5
        self.C6 = self.C5 + self.D6
        self.C7 = self.C6 + self.D7
        self.C8 = self.C7 + self.D8
        self.C9 = self.C8 + self.D9
        self.C10 = self.C9 + self.D10

        self.D = self.C10

        self.construct()

    def display_hidden_state(self, X: np.ndarray) -> None:
        print("---------")
        for i, x in enumerate(X):
            print(f"y_{i}: {self.m_inv[np.argmax(x[: self.C1])]}")
            print(f"p_{i}: {int(x[self.C1 + 1])}")
            if x[self.C2 : self.C3].sum() > 0:
                print(f"q↑_{i}: {self.s_inv[np.argmax(x[self.C2: self.C3])]}")
            print(f"c↑_{i}: {x[self.C3]}")
            print(f"l↑_{i}: {x[self.C4]}")
            if x[self.C5 : self.C6].sum() > 0:
                print(f"γ↑_{i}: {self.g_inv[np.argmax(x[self.C5: self.C6])]}")
            if x[self.C6 : self.C7].sum() > 0:
                print(f"q↓_{i}: {self.s_inv[np.argmax(x[self.C6: self.C7])]}")
            if x[self.C7 : self.C8].sum() > 0:
                print(f"γ↓_{i}: {self.g_inv[np.argmax(x[self.C7: self.C8])]}")
            if x[self.C8 : self.C9].sum() > 0:
                print(f"a↓_{i}: {self.a_inv[np.argmax(x[self.C8: self.C9])]}")
            if x[self.C9 : self.C10].sum() > 0:
                print(f"q↓, a↓, γ↓, y: {self.n_inv[np.argmax(x[self.C9: self.C10])]}")
            print("---------")

    def set_up_orderings(self):
        # Ordering of Q x A x Γ x Σ
        self.n = dict()
        self.n_inv = dict()
        for i, (q, a, γ, y) in enumerate(
            product(self.P.Q, self.P.actions, self.P.Γ, self.P.Σ)
        ):
            self.n[(q, a, γ, y)] = i
            self.n_inv[i] = (q, a, γ, y)

        # Ordering of Σbar
        self.m = {a: i for i, a in enumerate(self.SigmaEOS)}
        self.m_inv = {i: a for i, a in enumerate(self.SigmaEOS)}

        # Ordering of Q
        self.s = {q: i for i, q in enumerate(self.P.Q)}
        self.s_inv = {i: q for i, q in enumerate(self.P.Q)}

        # Ordering of Γ
        self.g = {γ: i for i, γ in enumerate(self.Gamma)}
        self.g_inv = {i: γ for i, γ in enumerate(self.Gamma)}

        # Ordering of A
        self.a = {a: i for i, a in enumerate(self.P.actions)}
        self.a_inv = {i: a for i, a in enumerate(self.P.actions)}

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
        elif component == "action":
            y = np.zeros((self.n_actions))
            y[self.a[x]] = 1
            return y
        else:
            raise TypeError
        return y

    def eq2q(self, x: np.ndarray) -> int:
        return self.s_inv[np.argmax(x[self.C2 : self.C3])]

    def ey2y(self, x: np.ndarray) -> str:
        return self.m_inv[np.argmax(x[: self.C1])]

    def initial_static_encoding(self, y: str, t: int) -> np.ndarray:
        X0 = np.concatenate(
            [
                self.one_hot(y, "symbol"),
                self.positional_encoding(t),
                # ----------------------------
                np.zeros(self.n_states),
                np.zeros(1),
                np.zeros(1),
                np.zeros(self.n_stack_symbols),
                # ----------------------------
                self.one_hot(self.q0, "state") if t == 0 else np.zeros(self.n_states),
                (
                    self.one_hot(BOT, "stack")
                    if t == 0
                    else np.zeros(self.n_stack_symbols)
                ),
                (
                    self.one_hot(Action.PUSH, "action")
                    if t == 0
                    else np.zeros(self.n_actions)
                ),  # It should be PUSH, because you imagine you "pushed" the BOT
                # self.one_hot(Action.PUSH, "action"),
                np.zeros(self.D10),
            ]
        )

        return X0

    def positional_encoding(self, t: int) -> np.ndarray:
        return np.asarray([1, t + 1, 1 / (t + 1), 1 / (t + 1) ** 2])

    def action_encoding(self, a: str) -> np.ndarray:
        if a == "PUSH":
            return np.asarray([1])
        elif a == "POP":
            return np.asarray([-1])
        elif a == "NOOP":
            return np.asarray([0])
        else:
            raise ValueError

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

    def construct_state_lookup_head(self) -> AttentionHead:
        """Constructs the transformer head that looks up the state the previous time
        step transitioned into, i.e., the current state.

        Returns:
            AttentionHead: The transformer head.
        """
        Wq = np.zeros((2, self.D))
        Wq[:, self.C1 : self.C1 + 2] = np.eye(2)
        bq = np.asarray([0, -1])

        Wk = np.zeros((2, self.D))
        P = np.zeros((2, 2))
        P[0, 1] = -1
        P[1, 0] = 1
        Wk[:, self.C1 : self.C1 + 2] = P

        # Copy the target state into the source state
        Wv = np.zeros((self.D, self.D))
        Wv[self.C2 : self.C3, self.C6 : self.C7] = np.eye(self.n_states)

        def Q(X):
            return (Wq @ X.T).T + bq

        def K(X):
            return (Wk @ X.T).T

        def V(X):
            return (Wv @ X.T).T

        def f(q, k):
            # print("111111111111111111111111111111111111111111")
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

    def construct_positions_written_head(self) -> AttentionHead:
        """Construct a transformer head that computes the *stack* positions
        of where each time step wrote to.

        Returns:
            AttentionHead: The transformer head.
        """
        Wq = np.zeros((1, self.D))
        Wk = np.zeros((1, self.D))
        Wv = np.zeros((self.D, self.D))
        Wv[self.C3, self.C8 + self.a[Action.PUSH]] = 1
        Wv[self.C3, self.C8 + self.a[Action.POP]] = -1

        def Q(X):
            return (Wq @ X.T).T

        def K(X):
            return (Wk @ X.T).T

        def V(X):
            return (Wv @ X.T).T

        def f(q, k):
            return -np.abs(np.dot(q, k.T))

        def O(X):  # noqa: E741, E743
            # print("2222222222222222222222222222222222222222")
            # print(X[self.C3])
            return X

        return AttentionHead(
            Q=Q,
            K=K,
            V=V,
            f=f,
            projection=ProjectionFunctions.averaging_hard,
            O=O,
        )

    def construct_stack_top_lookup_head(self) -> AttentionHead:
        """Construct a transformer head that computes the *string* position
        when the stack was written to last time.

        Returns:
            AttentionHead: The transformer head.
        """
        Wq = np.zeros((4, self.D))
        Wq[0, self.C3] = 1
        Wq[1, self.C1 + 2] = 1
        # Wq[2, self.C1 + 2] = 1
        # Wq[2, self.C1 + 3] = 1
        # Wq[2, self.C1 + 3] = 1 / 3
        bq = np.asarray([0, 0, 1, 1])

        Wk = np.zeros((4, self.D))
        Wk[0, self.C1 + 2] = 1
        Wk[1, self.C3] = -1
        # Wk[2, self.C1 + 2] = 1
        Wk[2, self.C1 + 3] = 1
        # Wk[2, self.C1 + 3] = 1
        Wk[3, self.C8 + self.a[Action.PUSH]] = 0
        Wk[3, self.C8 + self.a[Action.POP]] = -1

        Wv = np.zeros((self.D, self.D))
        Wv[self.C5 : self.C6, self.C7 : self.C8] = np.eye(self.n_stack_symbols)

        def Q(X):
            return (Wq @ X.T).T + bq

        def K(X):
            self.display_hidden_state(X)
            return (Wk @ X.T).T

        def V(X):
            return (Wv @ X.T).T

        def f(q, k):
            print("0000000000000000000000000000000000000")
            return (
                -np.abs(np.dot(q[:2], k[:2].T))
                - q[-2] * k[-2]
                - int(abs(q[1] - k[0]) < 1e-6)
                + q[-1] * k[-1]
            )
            # return -np.abs(np.dot(q, k.T))
            # return np.dot(q, k.T)

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

    def construct_layer_1(self):
        """Construct the parameters of the first transformer block."""

        H1 = self.construct_copy_head()
        H2 = self.construct_positions_written_head()
        heads = [H1, H2]

        Wh = np.zeros((self.D, len(heads) * self.D))
        # Copy over all existing information
        Wh[: self.D, : self.D] = np.eye(self.D)
        Wh[self.C3, self.C3] = 0
        # Copy over the new information about the stack writing positions
        Wh[self.C3, self.D + self.C3] = 1

        def fH(Z):
            print("--------------------------------------")
            # self.display_hidden_state(Z[:, : self.D])
            # print("--------------------")
            self.display_hidden_state(Z[:, self.D : 2 * self.D])
            print("--------------------------------------")
            return (Wh @ Z.T).T

        return MultiHeadAttentionLayer(heads=heads, fH=fH)

    def construct_transition_gates(self) -> Tuple[np.ndarray, np.ndarray]:
        W = np.zeros((self.D, 3 * self.D))
        b = np.zeros((self.D))
        for q, γ, y in product(self.P.Q, self.P.Γ, self.P.Σ):
            (qʼ, action, γʼ), _ = self.P.δ[q][y][γ]
            _w, _b = construct_and(
                3 * self.D,
                [
                    self.m[y],  # Symbol information from the copy head
                    # State information from the state head
                    self.D + self.C2 + self.s[q],
                    # Stack information from the stack head
                    2 * self.D + self.C5 + self.g[γ],
                ],
            )
            s = (qʼ, action, γʼ, y)
            W[self.C9 + self.n[s], :] = np.maximum(
                W[self.C9 + self.n[s], :], _w.reshape((-1,))
            )
            b[self.C9 + self.n[s]] = np.minimum(_b, b[self.C9 + self.n[s]])

        return W, b

    def construct_layer_2_combine_parameters(self):
        Wo, bo = self.construct_transition_gates()

        def transition(Z):
            # print(Wo[self.C9 + self.n[(0, Action.POP, "1", "a")], : self.D])
            # print(Wo[self.C9 + self.n[(0, Action.POP, "1", "a")], self.D : 2 * self.D])
            # print(Wo[self.C9 + self.n[(0, Action.POP, "1", "a")], 2 * self.D :])
            # print(Wo[self.C9 + self.n[(0, Action.POP, "1", "a")], self.m["a"]])
            # print(
            #     Wo[
            #         self.C9 + self.n[(0, Action.POP, "1", "a")],
            #         self.D + self.C2 + self.s[0],
            #     ]
            # )
            # print(
            #     Wo[
            #         self.C9 + self.n[(0, Action.POP, "1", "a")],
            #         2 * self.D + self.C5 + self.g["1"],
            #     ]
            # )
            # print(bo[self.C9 + self.n[(0, Action.POP, "1", "a")]])
            return ReLU((Wo @ Z.T).T + bo)

        # Check if the state is already present on the tape
        P = np.zeros((self.n_states, 3 * self.D))
        P[:, self.C6 : self.C7] = np.eye(self.n_states)
        # Produces a "mask" of 1s if the state is already present on the tape
        E = np.ones((self.D, self.n_states))

        # Decode the new state, action, and stack symbol from the transition head
        U = np.zeros((self.D, self.D))
        for q, a, γ, y in product(self.P.Q, self.P.actions, self.P.Γ, self.P.Σ):
            U[self.C6 + self.s[q], self.C9 + self.n[(q, a, γ, y)]] = 1
            U[self.C7 + self.g[γ], self.C9 + self.n[(q, a, γ, y)]] = 1
            U[self.C8 + self.a[a], self.C9 + self.n[(q, a, γ, y)]] = 1

        # Copy the old information
        # If some fields are empty, they will be filled in by the addition
        # If they are not empty, the added vectors will be zero
        Wh = np.zeros((self.D, 3 * self.D))
        Wh[: self.C9, : self.C9] = np.eye(self.C9)

        def combine(Z_original, Z_transitioned):
            multi_hot = H(U @ Z_transitioned.T).T
            mask = (E @ P @ Z_original.T).T
            masked_multi_hot = H(multi_hot - mask)

            return (Wh @ Z_original.T).T + masked_multi_hot

        return transition, combine

    def construct_layer_2(self):
        """Construct the parameters of the second transformer block."""

        H1 = self.construct_copy_head()
        H2 = self.construct_state_lookup_head()
        H3 = self.construct_stack_top_lookup_head()
        heads = [H1, H2, H3]

        transition, combine = self.construct_layer_2_combine_parameters()

        def fH(Z):
            print("--------------------------------------")
            self.display_hidden_state(Z[:, : self.D])
            print("--------------------")
            self.display_hidden_state(Z[:, self.D : 2 * self.D])
            print("--------------------")
            self.display_hidden_state(Z[:, 2 * self.D :])
            print("--------------------")
            print("Transitioned Z:")
            self.display_hidden_state(transition(Z))
            print("--------------------------------------")
            return combine(Z, transition(Z))

        return MultiHeadAttentionLayer(heads=heads, fH=fH)

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

        # Set up layers:
        MAH_1 = self.construct_layer_1()
        MAH_2 = self.construct_layer_2()

        # Set up the output matrix
        # E = self.construct_output_matrix()

        Wf = np.eye(self.D)

        def F(x):
            return (Wf @ x.T).T

        T = Transformer(
            layers=[MAH_1, MAH_2],
            F=F,
            encoding=self.one_hot,
            positional_encoding=self.positional_encoding,
            X0=self.initial_static_encoding,
            Tf=self,
        )

        self.T = T

        # self.lm = TransfomerLM(T, E)

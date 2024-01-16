from typing import Callable, List, Sequence

import numpy as np

from turnformer.base.symbols import EOS


class AttentionHead:
    """A class implementing the attention mechanism."""

    def __init__(
        self,
        Q: Callable[[np.ndarray], np.ndarray],
        K: Callable[[np.ndarray], np.ndarray],
        V: Callable[[np.ndarray], np.ndarray],
        f: Callable[[np.ndarray], np.ndarray],
        projection: Callable[[np.ndarray], np.ndarray],
        O: Callable[[np.ndarray], np.ndarray],  # noqa: E741, E743
    ):
        self.Q = Q
        self.K = K
        self.V = V

        self.f = f
        self.projection = projection

        self.O = O  # noqa: E741, E743

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Applies the attention mechanism to the input X, where the query is
        simply based on the last entry in X.

        Args:
            X (np.ndarray): The input to the attention mechanism.

        Returns:
            np.ndarray: The output of the attention mechanism.
        """

        # Q = self.Q(X)
        q = self.Q(X[-1, :])
        K = self.K(X)
        V = self.V(X)

        T = X.shape[0]
        s = self.projection(np.asarray([self.f(q, K[t, :]) for t in range(T)]))
        # print(f"s = {s}")
        a = np.dot(s, V)
        # print(f"X = {X}")
        # print(f"q = {q}")
        # print(f"K = {K}")
        # print(f"V = {V}")
        # print(f"a = {a}")

        # z = self.O(a) + a
        z = self.O(a)

        # print(f"z = {z}")
        # print()

        return z


class MultiHeadAttentionLayer:
    """A class implementing a single layer of the Transformer network based on the
    AttentionHead mechanism class.
    """

    def __init__(
        self,
        heads: List[AttentionHead],
        fH: Callable[[np.ndarray], np.ndarray],
    ):
        self.heads = heads
        self.fH = fH

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Applies the Transformer layer to the input X.

        Args:
            X (np.ndarray): The input to the Transformer layer.

        Returns:
            np.ndarray: The output of the Transformer layer.
        """

        T = X.shape[0]

        Zs = []
        for h, H in enumerate(self.heads):  # Iterate over the heads
            # Zs.append(np.vstack([H(X[: t + 1, :]) + X[t, :] for t in range(T)]))
            Zs.append(np.vstack([H(X[: t + 1, :]) for t in range(T)]))
            # print()

        Z = self.fH(np.hstack(Zs))

        return Z


class Transformer:
    """A class implementing the Transformer network."""

    def __init__(
        self,
        layers: Sequence[MultiHeadAttentionLayer],
        F: Callable[[np.ndarray], np.ndarray],
        encoding: Callable[[str], np.ndarray],
        positional_encoding: Callable[[int], np.ndarray],
        X0: Callable[[str], np.ndarray],
        Tf,
    ):
        self.layers = layers
        self.F = F
        self.encoding = encoding
        self.positional_encoding = positional_encoding
        self.X0 = X0
        self.Tf = Tf

    def __call__(self, y: str) -> np.ndarray:
        """Applies the Transformer to the input string y.

        Args:
            y (str): The input string.

        Returns:
            np.ndarray: The output of the Transformer layer.  # TODO
        """

        if len(y) == 0:
            return self.F(self.X0(y)).T

        X = self.X0(y[0], 0)
        X = X.reshape((-1, len(X)))
        # print(f"X0.shape = {X.shape}")
        # self.Tf.display_hidden_state(X)

        for t, yt in enumerate(y[1:]):
            X = np.vstack([X, self.X0(yt, t + 1)])

            # print(f"X{t + 1}.shape = {X.shape}")
            # self.Tf.display_hidden_state(X)

            Z = X
            for ll, layer in enumerate(self.layers):
                Z = layer(Z)

            X[-1, :] = Z[-1, :]
            # print("STATES")
            # for i in range(X.shape[0]):
            #     print(f"state {i}: {self.Tf.eq2q(X[i, :])}")
            # print("SYMBOLS")
            # for i in range(X.shape[0]):
            #     print(f"symbol {i}: {self.Tf.ey2y(X[i, :])}")
            # print()
            # print()

            # At this point, X[-1, :] should contain the new state (of the automaton)

        return self.F(X[-1, :]).T


class TransfomerLM:
    def __init__(self, T: Transformer, E: np.ndarray):
        self.T = T
        self.E = E

    def __call__(self, y: str) -> float:
        logp = 0
        for t, yt in enumerate(y):
            zt = self.T(y[: t + 1])
            # print(f"qt = {self.T.Tf.s_inv[np.argmax(zt)]}")
            logpt = (self.E[:, zt.argmax()])[self.T.encoding(yt).argmax()]
            # print(f"logpt = {logpt}")
            # print(f"pt = {np.exp(logpt)}")
            logp += logpt

        zt = self.T(y + y[-1])
        logpEOS = (self.E[:, zt.argmax()])[self.T.encoding(EOS).argmax()]
        # print(f"qT = {self.T.Tf.s_inv[np.argmax(zt)]}")
        # print(f"logpEOS = {logpEOS}")
        # print(f"pEOS = {np.exp(logpEOS)}")

        logp += logpEOS

        return logp

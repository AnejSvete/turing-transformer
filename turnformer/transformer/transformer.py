from typing import Callable, Sequence

import numpy as np

from rayuela.nn.rnn.utils import to_compatible_string


class Attention:
    """A class implementing the attention mechanism."""

    def __init__(
        self,
        Q: Callable[[np.ndarray], np.ndarray],
        K: Callable[[np.ndarray], np.ndarray],
        V: Callable[[np.ndarray], np.ndarray],
        f: Callable[[np.ndarray], np.ndarray],
        projection: Callable[[np.ndarray], np.ndarray],
    ):
        self.Q = Q
        self.K = K
        self.V = V

        self.f = f
        self.projection = projection

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
        print(f"s = {s}")
        a = np.dot(s, V)
        print(f"X = {X}")
        print(f"q = {q}")
        print(f"K = {K}")
        print(f"V = {V}")
        print(f"a = {a}")

        return a


class TransformerLayer:
    """A class implementing a single layer of the Transformer network based on the
    Attention mechanism class.
    """

    def __init__(
        self,
        Q: Callable[[np.ndarray], np.ndarray],
        K: Callable[[np.ndarray], np.ndarray],
        V: Callable[[np.ndarray], np.ndarray],
        f: Callable[[np.ndarray], np.ndarray],
        projection: Callable[[np.ndarray], np.ndarray],
        O: Callable[[np.ndarray], np.ndarray],  # noqa: E741, E743
    ):
        self.A = Attention(Q, K, V, f, projection)
        self.O = O  # noqa: E741, E743

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Applies the Transformer layer to the input X.

        Args:
            X (np.ndarray): The input to the Transformer layer.

        Returns:
            np.ndarray: The output of the Transformer layer.
        """

        T = X.shape[0]

        # ! I think this simulates the whole execution of the automaton
        # ! at every time step, so this could be made more efficient
        # A = np.vstack([self.A(X[: t + 1, :]) + X[t, :] for t in range(T)])
        A = np.vstack([self.A(X[: t + 1, :]) for t in range(T)])

        # Z = self.O(A) + A
        Z = self.O(A)

        return Z


class Transformer:
    """A class implementing the Transformer network."""

    def __init__(
        self,
        layers: Sequence[TransformerLayer],
        F: Callable[[np.ndarray], np.ndarray],
        encoding: Callable[[str], np.ndarray],
        positional_encoding: Callable[[int], np.ndarray],
        S: np.ndarray,
        Tf,
    ):
        self.layers = layers
        self.F = F
        self.encoding = encoding
        self.positional_encoding = positional_encoding
        self.S = S
        self.Tf = Tf

    def __call__(self, y: str) -> np.ndarray:
        """Applies the Transformer to the input string y.

        Args:
            y (str): The input string.

        Returns:
            np.ndarray: The output of the Transformer layer.  # TODO
        """

        X = np.concatenate([self.encoding(y[0]), self.positional_encoding(0), self.S])
        print(f"Xi.shape = {X.shape}")
        print(f"Xi = {X}")

        for t, yt in enumerate(to_compatible_string(y[1:])):
            X = np.vstack(
                [
                    X,
                    np.concatenate(
                        [
                            self.encoding(yt),
                            self.positional_encoding(t + 1),
                            np.zeros((self.S.shape[0])),
                        ]
                    ),
                ]
            )

            print(f"X{t}.shape = {X.shape}")
            print(f"X{t} = {X}")

            Z = X
            for ll, layer in enumerate(self.layers):
                print(f"Layer {ll}")
                Z = layer(Z)
                if ll == 0:  # TODO
                    P = np.vstack([self.positional_encoding(i) for i in range(t + 2)])
                    Z = np.hstack([Z, P])

            X[-1, -self.S.shape[0] :] = Z[-1, :]
            print("STATES")
            for i in range(X.shape[0]):
                print(f"state {i}: {self.Tf.eq2q(X[i, -self.S.shape[0] :])}")
            print("SYMBOLS")
            for i in range(X.shape[0]):
                print(f"symbol {i}: {self.Tf.ey2y(X[i, : self.Tf.n_symbols + 1])}")
            print()
            print()

            # At this point, X[-1, :] should contain the new state (of the automaton)

        return self.F(X[-1, :])


class TransfomerLM:
    def __init__(self, T: Transformer, E: np.ndarray):
        self.T = T
        self.E = E

    def __call__(self, y: str) -> float:
        ...

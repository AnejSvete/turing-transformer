import sympy as sp
from sympy import Rational, RealNumber


def sp_max(x: sp.Matrix) -> Rational:
    return Rational(sp.Max(*x))


def sp_argmax(x: sp.Matrix) -> sp.Matrix:
    return sp.Matrix([i for i, _ in enumerate(x) if x[i] == sp_max(x)])


def ρ(x: sp.Matrix) -> sp.Matrix:
    """Implements the hardmax function, which is defined as being 1/r at index d if the
    maximum value of x is at index d and there are d indices that reach the maximum
    value, and 0 otherwise.

    Returns:
        Matrix: The hardmax vector.
    """
    am = sp_argmax(x)
    if len(am) == 1:
        return sp.Matrix(
            [Rational(1) if i == am[0] else Rational(0) for i in range(len(x))]
        )
    else:
        return sp.Matrix(
            [Rational(1) / len(am) if i in am else Rational(0) for i in range(len(x))]
        )


def _compute_scores(q: sp.Matrix, K: sp.Matrix) -> sp.Matrix:
    scores = sp.zeros(1, K.shape[0])
    for i in range(K.shape[0]):
        # scores[0, i] = Rational(-1) * Rational(sp.Abs(q.dot(K[i, :])))
        scores[0, i] = Rational(q.dot(K[i, :]))
    return scores


def single_query_attention(q: sp.Matrix, K: sp.Matrix, V: sp.Matrix) -> sp.Matrix:
    scores = _compute_scores(q, K)
    s = ρ(scores)
    a = s.T @ V
    return a


def attention(Q: sp.Matrix, K: sp.Matrix, V: sp.Matrix) -> sp.Matrix:
    A = sp.zeros(Q.shape[0], V.shape[1])
    for i in range(Q.shape[0]):
        A[i, :] = single_query_attention(Q[i, :], K, V)
    return A


def test():
    import numpy as np

    K, V = sp.Matrix(np.random.rand(3, 2), dtype=Rational), sp.Matrix(
        np.random.rand(3, 2), dtype=Rational
    )
    q = sp.Matrix(np.random.rand(1, 2), dtype=Rational)

    sp.pprint(q)
    sp.pprint(K)
    sp.pprint(V)

    a = single_query_attention(q, K, V)

    sp.pprint(a)


def test_single_query_attention(D: int = 3, N: int = 5):
    """Tests the single query attention function. This assumes the dot-product score."""
    import numpy as np

    for _ in range(100):
        q = np.random.rand(1, D)
        q = q / np.linalg.norm(q)
        K = np.random.rand(N, D)
        K = K / np.linalg.norm(K, axis=1).reshape(-1, 1)
        V = np.random.rand(N, D)

        k = np.random.randint(1, N)
        idxs = list(np.random.choice(N, k, replace=False))
        for idx in idxs:
            K[idx, :] = q

        q = sp.Matrix(q, dtype=Rational)
        K = sp.Matrix(K, dtype=Rational)
        V = sp.Matrix(V, dtype=Rational)

        a = single_query_attention(q, K, V)

        assert (
            np.asarray(a).flatten() - np.sum(np.asarray(V[idxs, :]), axis=0) / k
        ).dot(
            np.asarray(a).flatten() - np.sum(np.asarray(V[idxs, :]), axis=0) / k
        ) < 1e-6


test_single_query_attention()

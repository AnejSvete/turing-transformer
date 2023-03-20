from turnformer.base.symbol import Sym
from turnformer.base.string import String
from turnformer.base.alphabet import Alphabet
from turnformer.base.turing_machine import TuringMachine

from sympy import Rational
import sympy as sp


class Embedding:
    def __init__(self, T: TuringMachine) -> None:
        assert T.frozen
        self.T = T
        self.Γ = T.Γ
        self.Q = T.Q
        self.D = 4 * len(self.Γ) + 2 * len(self.Q) + 11
        # Partitions of the embedding space:
        # q_1, s_1, x_1,                                      |Q| + |Γ| + 1
        # q_2, s_2, x_2, x_3, x_4, x_5,                       |Q| + |Γ| + 4
        # s_3, x_6, s_4, x_7,                                 |Γ| + 2
        # x_8, x_9, x_10, x_11]                               4
        self.sym2idx = {sym: i for i, sym in enumerate(self.Γ)}

    def __call__(self, y: String) -> sp.Matrix:
        X = sp.zeros(self.D, len(y), dtype=Rational(0))
        for n, sym in enumerate(y):
            X[:, n] = self.embed_symbol(sym, n + 1)
        return X

    def embed_symbol(self, sym: Sym, n: int) -> sp.Matrix:
        v = sp.zeros(self.D, 1, dtype=Rational(0))
        v[
            self.offset(component=3) + self.sym2idx[sym],
            0,
        ] = Rational(1)
        v = v + self.position_encoding(n)
        return v

    def offset(self, component: int) -> int:
        if component == 1:
            return 0
        elif component == 2:
            return len(self.Q) + len(self.Γ) + 1
        elif component == 3:
            return len(self.Q) + len(self.Γ) + 1 + len(self.Q) + 2 * len(self.Γ) + 4
        elif component == 4:
            return (
                len(self.Q)
                + len(self.Γ)
                + 1
                + len(self.Q)
                + 2 * len(self.Γ)
                + 4
                + len(self.Γ)
                + 2
            )

        raise ValueError(f"Invalid component: {component}")

    def position_encoding(self, n: int) -> sp.Matrix:
        v = sp.zeros(self.D, 1, dtype=Rational(0))
        v[self.offset(component=4) + 1, 0] = Rational(n)
        v[self.offset(component=4) + 2, 0] = Rational(f"1/{n}")
        v[self.offset(component=4) + 3, 0] = Rational(f"1/{n**2}")
        return v


def test():
    from turnformer.base.turing_machine import random_turing_machine

    T = random_turing_machine(2, Alphabet({Sym("a"), Sym("b")}), p=0.9)
    embedding = Embedding(T)

    for y in [String("aaa"), String("aba"), String("bba")]:
        X = embedding(y)
        sp.pprint(X)
        print("\n\n\n\n------------------------\n\n\n\n")

from typing import Set, Dict, Tuple, List, Optional, Union, Generator
from enum import IntEnum, unique
from pprint import pformat
from itertools import product


import numpy as np
from frozendict import frozendict


from turnformer.base.symbol import Sym, EMPTY
from turnformer.base.string import String
from turnformer.base.alphabet import Alphabet
from turnformer.base.state import State


RandomGenerator = Union[np.random.Generator, int]


@unique
class Direction(IntEnum):
    LEFT = -1
    RIGHT = 1

    def __str__(self) -> str:
        return "L" if self == Direction.LEFT else "R"

    def __repr__(self) -> str:
        return "L" if self == Direction.LEFT else "R"


class TuringMachine:
    def __init__(self, Σ: Alphabet) -> None:
        self.Σ = Σ
        self.Γ = Alphabet([EMPTY, *Σ.symbols])
        self.sym2idx = {sym: i for i, sym in enumerate(self.Γ)}  # type: Dict[Sym, int]
        self.Q = set()  # type: Union[frozenset[State], Set[State]]
        self.q2idx = dict()  # type: Dict[State, int]
        self.qsym2idx = dict()  # type: Dict[Tuple[State, Sym], int]
        self.qsymd2idx = dict()  # type: Dict[Tuple[State, Sym, Direction], int]
        self.δ = dict()  # type: Dict[Tuple[State, Sym], Tuple[State, Sym, Direction]]
        self.q_init = None  # type: Optional[State]
        self.F = set()  # type: Union[frozenset[State], Set[State]]
        self.frozen = False

    def freeze(self) -> None:
        """Freezes the Turing machine to disable further modifications."""
        self.Q = frozenset(self.Q)
        self.F = frozenset(self.F)
        self.δ = frozendict(self.δ.items())
        self.q2idx = {q: i for i, q in enumerate(self.Q)}
        self.qsym2idx = {
            (q, a): self.sym2idx[a] * len(self.Q) + self.q2idx[q]
            for (q, a) in product(self.Q, self.Γ.symbols)
        }
        self.qsymd2idx = {
            (q, a, d): self.sym2idx[a] * len(self.Q)
            + self.q2idx[q] * 2
            + (d == Direction.LEFT)
            for (q, a, d) in product(
                self.Q, self.Γ.symbols, [Direction.LEFT, Direction.RIGHT]
            )
        }
        self.frozen = True

    def add_transition(
        self,
        q: State,
        a: Sym,
        q_next: State,
        b: Sym,
        d: Direction,
    ) -> None:
        """Adds a transition to the Turing machine.

        Args:
            q (State): The current state.
            a (Sym): The current symbol.
            q_next (State): The next state.
            b (Sym): The next symbol.
            d (Direction): The direction of the head.
        """
        assert a in self.Σ, f"Symbol {a} not in Σ."
        assert b in self.Γ, f"Symbol {b} not in Γ."
        assert not self.frozen
        self.Q.add(q)
        self.Q.add(q_next)
        self.δ[(q, a)] = (q_next, b, d)

    def add_initial_state(self, q: State) -> None:
        """Adds an initial state to the Turing machine.

        Args:
            q (State): The initial state.
        """
        assert not self.frozen
        self.Q.add(q)
        self.q_init = q

    def add_final_state(self, q: State) -> None:
        """Adds a final state to the Turing machine.

        Args:
            q (State): The final state.
        """
        assert not self.frozen
        self.Q.add(q)
        self.F.add(q)

    def add_final_states(self, qs: List[State]) -> None:
        """Adds a list of final states to the Turing machine.

        Args:
            qs (List[State]): The list of final states.
        """
        assert not self.frozen
        for q in qs:
            self.add_final_state(q)

    def transitions(
        self, q: State
    ) -> Generator[Tuple[State, Sym, State, Sym, Direction], None, None]:
        """Returns the transitions of a state.

        Args:
            q (State): The state.

        Yields:
            Tuple[State, Sym, State, Sym, Direction]: The transitions.
        """
        for a in self.Γ:
            if (q, a) in self.δ:
                q_next, b, d = self.δ[(q, a)]
                yield q, a, q_next, b, d

    def __call__(self, y: String, verbose: bool = False) -> bool:
        """Runs the Turing machine on a string.

        Args:
            y (String): The input string.

        Returns:
            bool: Whether the string is accepted by the Turing machine.
        """
        tape = y.y + [EMPTY]
        head = 0
        q = self.q_init
        while q not in self.F:
            a = tape[head]
            if (q, a) not in self.δ:
                if verbose:
                    print(f"δ({q}, {a}) not defined.")
                return False
            q_next, b, d = self.δ[(q, a)]
            if verbose:
                print(f"(q, a) = ({q}, {a}) -> (q_next, b, d) = ({q_next}, {b}, {d})")
            tape[head] = b
            head += d if head + d >= 0 else 0
            q = q_next
        return q in self.F

    def __str__(self) -> str:
        return (
            f"Q: {self.Q}\n\n Σ: {self.Σ}\n\n Γ: {self.Γ}\n\n δ: {pformat(self.δ)}\n\n "
            + f"q_init: {self.q_init}\n\n F: {self.F}"
        )


def random_turing_machine(
    n_states: int,
    Σ: Alphabet,
    p: float = 0.5,
    p_final: float = 0.1,
    rng: RandomGenerator = 42,
) -> TuringMachine:
    """Generates a random Turing machine.

    Args:
        n_states (int): The number of states in the Turing machine.
        p (float): The average number of transitions per state.
        Σ (Alphabet): The input alphabet.

    Returns:
        TuringMachine: The random Turing machine.
    """
    rng = np.random.default_rng(rng)

    tm = TuringMachine(Σ)
    for i in range(n_states):
        q = State(i)
        for a in Σ:
            if rng.random() > p:
                continue

            q_next = State(rng.integers(n_states))
            b = tm.Γ[rng.integers(len(tm.Γ))]
            d = Direction.LEFT if rng.random() < 0.5 else Direction.RIGHT
            tm.add_transition(q, a, q_next, b, d)

    tm.add_initial_state(State(0))

    for i in range(n_states - 1):
        if rng.random() < p_final:
            tm.add_final_state(State(i))

    tm.add_final_state(State(n_states - 1))

    return tm


def test():
    from turnformer.base.symbol import A, B

    T = random_turing_machine(10, Alphabet({A, B}), p=0.5, p_final=0.1, rng=42)

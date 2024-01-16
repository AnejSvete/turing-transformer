from enum import IntEnum, unique
from itertools import product
from math import log
from typing import List, Tuple

import numpy as np

from turnformer.base.symbols import BOT


@unique
class Action(IntEnum):
    PUSH = 0
    POP = 1
    NOOP = 2

    def __str__(self):
        action_names = {Action.PUSH: "PUSH", Action.POP: "POP", Action.NOOP: "NOOP"}
        return action_names[self]


class SingleStackPDA:
    def __init__(
        self,
        Σ: list = ["a", "b"],
        Γ: list = [BOT, "0", "1"],
        n_states: int = 1,
        seed: int = 42,
        randomize: bool = True,
    ):
        self.seed = seed

        self.Σ = Σ
        self.Γ = Γ
        self.Q = list(range(n_states))

        # δ: Q × Σ × Γ → ((Q × {PUSH, POP, NOOP} × Γ) × R)
        self.δ = {q: {sym: {γ: {} for γ in self.Γ} for sym in self.Σ} for q in self.Q}
        if randomize:
            self._random_δ()

    def step(self, q: int, stack: List[str], a: str) -> Tuple[List[str], float]:
        γ = stack[-1]
        qʼ, action, γʼ = self.δ[q][a][γ][0]

        # Modify the stack according to the action.
        if action == Action.PUSH:
            stack.append(γʼ)
        elif action == Action.POP:
            stack.pop()
        elif action == Action.NOOP:
            pass
        else:
            raise Exception

        # Return the new configuration and the probability of the action.
        return qʼ, stack, self.δ[q][a][γ][1]

    @property
    def probabilistic(self):
        """Checks if the PDA is probabilistic."""
        d = {(q, γ): 0 for q, γ in product(self.Q, self.Γ)}
        for q, a, γ in product(self.Q, self.Σ, self.Γ):
            d[(q, γ)] += self.δ[q][a][γ][1]
        return all(abs(d[(q, γ)] - 1) < 1e-6 for (q, γ) in product(self.Q, self.Γ))

    def visualize(self) -> str:
        """Produces an ASCII visual representation of the probabilistic PDA."""
        output = ""

        # Header
        output += "Probabilistic PDA\n"
        output += "-----------------\n\n"

        # States
        output += "States:\n"
        for q in self.Q:
            output += f"{q} "
        output += "\n"

        # Alphabet
        output += "Alphabet:\n"
        for a in self.Σ:
            output += f"{a} "
        output += "\n\n"

        # Stack alphabet
        output += "Stack Alphabet:\n"
        for γ in self.Γ:
            output += f"{γ} "
        output += "\n\n"

        # Transition probabilities
        output += "Transition Probabilities:\n"
        for q, γ, a in product(self.Q, self.Γ, self.Σ):
            (qʼ, action, γʼ), p = self.δ[q][a][γ]
            output += f"q{q} -({a}, {γ})→ q{qʼ} ({action}, {γʼ}): {p:.3f}\n"
        output += "\n"

        # Probabilistic PDA status
        output += "Probabilistic PDA Status:\n"
        output += f"Is Probabilistic: {self.probabilistic}\n"

        return output

    def accept(self, y: str) -> Tuple[bool, float]:
        """Computes the acceptance probability of a string. Returns a tuple of
        the acceptance status and the log probability.

        Args:
            y (str): The string to be accepted.

        Returns:
            Tuple[bool, float]: The acceptance status and the log probability.
        """

        stack = [BOT]
        logp = 0
        q = 0

        # Simulate a run of the PDA. (Assumes that the PDA is deterministic.)
        for a in y:
            q, stack, p = self.step(q, stack, a)
            logp += log(p)

        return stack == [BOT], logp

    def __call__(self, y: str) -> Tuple[bool, float]:
        """Computes the acceptance probability of a string. Returns a tuple of
        the acceptance status and the log probability.
        It simply calls the accept method.

        Args:
            y (str): The string to be accepted.

        Returns:
            Tuple[bool, float]: The acceptance status and the log probability.
        """
        return self.accept(y)

    def _random_δ(self):
        """Initializes a random transition function and with it a random PPDA."""
        rng = np.random.default_rng(self.seed)

        pushes = list((Action.PUSH, γ) for γ in self.Γ if γ != BOT)

        for q, γ in product(self.Q, self.Γ):
            # The possible actions have to form a probability distribution
            # for every (q, γ).
            α = rng.random(len(self.Σ))
            α /= α.sum()
            for ii, a in enumerate(self.Σ):
                qʹ = rng.integers(0, len(self.Q))
                if γ == BOT:
                    flip = rng.integers(0, 1, endpoint=True)
                    if flip == 0:
                        self.δ[q][a][γ] = ((qʹ, Action.NOOP, γ), α[ii])
                    else:
                        self.δ[q][a][γ] = ((qʹ,) + tuple(rng.choice(pushes)), α[ii])
                else:
                    flip = rng.integers(0, 2, endpoint=True)
                    if flip == 0:
                        self.δ[q][a][γ] = ((qʹ, Action.NOOP, γ), α[ii])
                    elif flip == 1:
                        self.δ[q][a][γ] = ((qʹ, Action.POP, γ), α[ii])
                    else:
                        self.δ[q][a][γ] = ((qʹ,) + tuple(rng.choice(pushes)), α[ii])


class TwoStackPDA:
    def __init__(
        self,
        Σ={"a", "b"},
        Γ_1={BOT, "0", "1"},
        Γ_2={BOT, "0", "1"},
        seed: int = 42,
        randomize: bool = True,
    ):
        self.seed = seed

        self.Σ = Σ
        self.Γ_1 = Γ_1
        self.Γ_2 = Γ_2

        # δ: Σ × (Γ_1 × Γ_2) → (({PUSH, POP, NOOP} × Γ_1 × Γ_2) × R)
        self.δ = {
            sym: {(γ_1, γ_2): {} for (γ_1, γ_2) in product(self.Γ_1, self.Γ_2)}
            for sym in self.Σ
        }
        if randomize:
            self._random_δ()

    def _execute_action(self, stack: List[str], action: Action, γ_new: str):
        """Commits an action to a the current stack configuration."""
        if action == Action.PUSH:
            stack.append(γ_new)
        elif action == Action.POP:
            stack.pop()
        elif action == Action.NOOP:
            pass
        else:
            raise Exception

    def step(
        self, stacks: Tuple[List[str], List[str]], a: str
    ) -> Tuple[Tuple[List[str], List[str]], float]:
        """Executes a step of the PDA. Returns a tuple of the new stacks and the
        probability of the action.

        Args:
            stacks (Tuple[List[str], List[str]]): The current stacks.
            a (str): The current symbol.

        Returns:
            Tuple[Tuple[List[str], List[str]], float]: The new stacks and the
                probability of the action.
        """
        assert a in self.Σ

        γ_1, γ_2 = stacks[0][-1], stacks[1][-1]
        (action_1, γ_1ʼ), (action_2, γ_2ʼ) = self.δ[a][(γ_1, γ_2)][:2]

        self._execute_action(stacks[0], action_1, γ_1ʼ)
        self._execute_action(stacks[1], action_2, γ_2ʼ)

        return stacks, self.δ[a][(γ_1, γ_2)][2]

    def accept(self, y: str) -> Tuple[bool, float]:
        """Computes the acceptance probability of a string. Returns a tuple of
        the acceptance status and the log probability.

        Args:
            y (str): The string to be accepted.

        Returns:
            Tuple[bool, float]: The acceptance status and the log probability.
        """

        stacks = [BOT], [BOT]
        logp = 0

        for a in y:
            stacks, p = self.step(stacks, a)
            logp += log(p)

        return stacks[0] == [BOT] and stacks[1] == [BOT], logp

    def __call__(self, y: str) -> Tuple[bool, float]:
        """Computes the acceptance probability of a string. Returns a tuple of
        the acceptance status and the log probability.
        It simply calls the accept method.

        Args:
            y (str): The string to be accepted.

        Returns:
            Tuple[bool, float]: The acceptance status and the log probability.
        """
        return self.accept(y)

    def _get_action(
        self, γ_top: str, pushes: List[Tuple[Action, str]], rng: np.random.Generator
    ):
        """Returns a random action and a random symbol for a given stack top."""
        if γ_top == BOT:
            flip = rng.integers(0, 1, endpoint=True)
            if flip == 0:
                action, γ_new = (Action.NOOP, γ_top)
            else:
                action, γ_new = tuple(rng.choice(pushes))
        else:
            flip = rng.integers(0, 2, endpoint=True)
            if flip == 0:
                action, γ_new = (Action.NOOP, γ_top)
            elif flip == 1:
                action, γ_new = (Action.POP, γ_top)
            else:
                action, γ_new = tuple(rng.choice(pushes))

        return action, γ_new

    def _random_δ(self):
        """Initializes a random transition function and with it a random PPDA."""
        rng = np.random.default_rng(self.seed)

        pushes = [
            [(Action.PUSH, γ) for γ in Γ] for Γ in [self.Γ_1 - {BOT}, self.Γ_2 - {BOT}]
        ]

        for γ_1, γ_2 in product(self.Γ_1, self.Γ_2):
            α = rng.dirichlet(np.ones(len(self.Σ)))
            for ii, a in enumerate(self.Σ):
                action_1, γ_1ʼ = self._get_action(γ_1, pushes[0], rng)
                action_2, γ_2ʼ = self._get_action(γ_2, pushes[1], rng)

                self.δ[a][(γ_1, γ_2)] = (action_1, γ_1ʼ), (action_2, γ_2ʼ), α[ii]

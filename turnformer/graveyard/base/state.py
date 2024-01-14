from typing import Any


class State:
    def __init__(self, idx: int) -> None:
        self.idx = idx

    def __str__(self) -> str:
        return f"q{self.idx}"

    def __repr__(self) -> str:
        return f"q{self.idx}"

    def __hash__(self) -> int:
        return hash(self.idx)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, State) and self.idx == other.idx

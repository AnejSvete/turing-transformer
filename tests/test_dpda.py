from turnformer.automata.pda import SingleStackPDA
from turnformer.base.symbols import BOT
from turnformer.transform.context_free import ContextFreeTransform


def test_unweighted():
    P = SingleStackPDA(
        Σ=["a", "b"],
        Γ=[BOT, "0", "1"],
        n_states=2,
        seed=42,
        randomize=True,
    )

    T = ContextFreeTransform(P)

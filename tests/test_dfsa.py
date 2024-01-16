import random

import numpy as np
from rayuela.fsa.random import random_pfsa

from turnformer.transform.finite_state import FiniteStateTransform


def test_dfsa():
    for _ in range(10):
        n_states = random.randint(1, 20)
        A = random_pfsa(
            Sigma="abcde",
            num_states=n_states,
            bias=0.4,
            deterministic=True,
        )

        T = FiniteStateTransform(A)

        for _ in range(100):
            length = random.randint(1, 10)
            y = "".join(random.choices(["a", "b", "c"], k=length))

            pA = A(y).value
            if pA == 0:
                continue

            logpA = np.log(A(y).value)

            logpT = T.lm(y)

            assert np.isclose(logpA, logpT, atol=1e-5)

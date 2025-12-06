"""
prepare_strain_random_walk.py

Generates synthetic multiaxial strain paths using a random-walk scheme.
Each of the six Voigt components evolves through random signed increments
to produce a broad, path-dependent strain dataset for constitutive modeling.
"""
import numpy as np

VOIGT6 = ("xx", "yy", "zz", "yz", "zx", "xy")

def prepare_strain_random_walk(m: int = 1000, T: int = 100, 
                               inc_min: float = 1e-6, inc_max: float = 1e-4, 
                               seed: int | None = None,):
    """
    Return (eps, deps) with shape (m, T, 6).
    """
    assert inc_max >= inc_min >= 0.0
    rng = np.random.default_rng(seed)

    signs = rng.choice([-1.0, 1.0], size=(m, T, 6))
    mags  = rng.uniform(inc_min, inc_max, size=(m, T, 6))
    deps  = signs * mags

    init_state = np.zeros((m, 1, 6))  
    deps = np.concatenate([init_state, deps], axis=1)

    eps   = np.cumsum(deps, axis=1)
    assert eps.shape == (m, T+1, 6) and deps.shape == (m, T+1, 6), \
        f"Unexpected output shape: eps={eps.shape}, deps={deps.shape}, expected ({m}, {T+1}, 6)"

    return eps, deps

if __name__ == "__main__":
    e, d = prepare_strain_random_walk(m=3, T=4, seed=0)
    assert e.shape == (3, 4, 6) and d.shape == (3, 4, 6)
    assert np.allclose(e[:, 0], d[:, 0])
    assert np.allclose(e[:, 1], d[:, 0] + d[:, 1])
    print("prepare_strain_random_walk sanity OK.")

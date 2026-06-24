"""
env_configs.py
--------------
Per-environment overrides for algorithm hyperparameters.

The default values are used for all environments not listed here.
Any key present in an environment's entry overrides the corresponding default.

Keys
----
bc_steps        : gradient steps for Algorithm 6 (BC)
dagger_pi_steps : gradient steps per outer iteration for Algorithm 7 (DAgger)
k_alg5          : number of outer iterations for Algorithm 5 (SPOIL)
q_lr            : learning rate for the Q-network
q_steps         : gradient steps per Q update
"""

_DEFAULTS = {
    "bc_steps":        100, 
    "dagger_pi_steps": 10,
    "k_alg5":          100,
    "q_lr":            1e-3,
    "q_steps":         10,
}

_ENV_OVERRIDES = {
    "CartPole-v1": {
        "dagger_pi_steps": 20,
    },
    "Pendulum-v1": {
        "bc_steps":        1500,
        "dagger_pi_steps": 50,
        "k_alg5":          200,
        "q_lr":            1e-3,
        "q_steps":         10,
    },
}



def get_env_config(env_name: str) -> dict:
    """
    Return the full config dict for the given environment,
    merging defaults with any per-environment overrides.
    """
    cfg = dict(_DEFAULTS)
    cfg.update(_ENV_OVERRIDES.get(env_name, {}))
    return cfg

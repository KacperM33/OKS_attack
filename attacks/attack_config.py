"""
Global hyperparameter configuration for all attacks.

Hyperparameters:
    eps: Step size in the forward perturbation (larger value = stronger effect).
    delta: Scale of the orthogonal perturbation (larger value = stronger effect).
    N: Number of sampled candidates.
    K: Number of iterations per frame.
    weight: How much the clean sample retains the previous perturbation
    para_rate: Score importance weight in the [0, 1] range (lower = temporal features are more important, higher = spatial features are more important).
    perturb_max: RMSE-norm limit (maximum allowed difference between images).
    decay: Retention rate of the previous perturbation (larger value = retains more of the previous noise).
    noise_scale: Strength of the initial noise.
"""

ATTACK_PARAMS = {
    'N': 30,               # deafult 5 
    'K': 15,               # deafult 3
    'eps': 2.0,            # deafult 1.0
    'delta': 10.0,         # deafult 0.1
    'weight': 0.9,         # deafult 0.9 
    'decay': 0.5,          # deafult 0.8
    'perturb_max': 10.0,   # deafult 5.0
    'para_rate': 0.5,      # deafult 0.9
    'noise_scale': 20.0    # deafult 20.0
}

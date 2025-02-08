import numpy as np
from math import log2, sqrt
from scipy.special import erfcinv

def normal_approximation(N, C, V, epsilon):
    """
    Returns an estimate of the number of information bits K for a blocklength N,
    channel capacity C (bits per channel use), channel dispersion V,
    and target block-error probability epsilon, via the normal approximation:
    
        K ≈ N*C - sqrt(N*V)*Q^-1(epsilon) + 0.5*log2(N)
    
    where Q^-1 is the inverse of the Q-function.

    Parameters
    ----------
    N : int
        Blocklength (number of channel uses).
    C : float
        Channel capacity in bits/channel use.
    V : float
        Channel dispersion in bits^2/channel use.
    epsilon : float
        Target block-error probability (0 < epsilon < 1).

    Returns
    -------
    float
        Normal approximation estimate for K.
    """

    # The Q-function is Q(x) = 0.5 * erfc(x / sqrt(2)).
    # Hence Q^-1(p) = sqrt(2) * erfcinv(2*p).
    Qinv = sqrt(2) * erfcinv(2.0 * epsilon)

    # Apply the normal approximation formula
    K_est = N * C - sqrt(N * V) * Qinv - 0.5 * log2(N)

    return K_est


# Example usage for a BSC with crossover probability p:
def binary_symmetric_channel_capacity(p):
    """
    Returns the channel capacity (C) in bits/use and dispersion (V) for a BSC(p).
    """
    from math import log2
    # Binary entropy function h2
    def h2(x):
        return -x*log2(x) - (1-x)*log2(1-x) if 0 < x < 1 else 0.0

    C = 1.0 - h2(p)
    
    # Dispersion V = p(1-p) [log2((1-p)/p)]^2
    if p in (0, 1):
        V = 0.0
    else:
        V = p * (1 - p) * ( (np.log2((1-p)/p)) ** 2 )

    return C, V


if __name__ == "__main__":
    # Example: BSC with p=0.1, blocklength N=1024, target BLER epsilon=1e-3
    p = 0.1
    N = 4096*8
    epsilon = 1e-4

    C, V = binary_symmetric_channel_capacity(p)
    K_est = normal_approximation(N, C, V, epsilon)
    print(f"Estimated K for BSC(p={p}), N={N}, epsilon={epsilon}: {K_est:.2f}")

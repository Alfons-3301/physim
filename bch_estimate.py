def compute_bch_nearest_rate(desired_rate):
    """
    Computes the nearest possible (n, k, t) triplets for a given desired rate.
    """
    import math
    possible_values = []
    
    # Consider a range of m values (practical BCH codes)
    for m in range(3, 11):  # m determines n = 2^m - 1
        n = 2**m - 1
        
        # Iterate over possible t values
        for t in range(1, (n // m) + 1):
            r = min(m * t, n)  # Parity bits
            k = n - r
            rate = k / n
            
            # Store the closest possible rate matches
            possible_values.append((n, k, t, rate))
    
    # Sort by the closest match to the desired rate
    possible_values.sort(key=lambda x: abs(x[3] - desired_rate))
    
    return possible_values[:10]  # Return the top 10 closest matches

# Example: Desired rate input
desired_rate = 0.2  # Example target rate
nearest_pairs = compute_bch_nearest_rate(desired_rate)

# Display results
import pandas as pd
df = pd.DataFrame(nearest_pairs, columns=["n", "k", "t", "Rate"])

print(df)

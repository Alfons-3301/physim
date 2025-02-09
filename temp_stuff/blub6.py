import numpy as np
import matplotlib.pyplot as plt

# Import your codecs and block parser.
from codec.errorCodecs import ReedSolomonCodec, PolarCodec
from codec.privacyAmp import PrivacyAmplification2
from util.parser import BlockParser

# =============================================================================
# Binary Symmetric Channel (BSC) Implementation
# =============================================================================
# (Assuming AbstractDiscreteChannel is defined in your project.)
class BinarySymmetricChannel:
    def __init__(self, error_probability: float):
        self.error_probability = error_probability

    def transmit(self, input_bits: np.ndarray) -> np.ndarray:
        # Flip each bit with probability equal to error_probability.
        noise = np.random.binomial(1, self.error_probability, size=input_bits.shape)
        return (input_bits + noise) % 2  # XOR-style flipping

    def set_parameters(self, params: dict):
        self.error_probability = params.get("error_probability", self.error_probability)

# =============================================================================
# Simulation Functions (Using BSC and No Modulation)
# =============================================================================

def simulate_ber_with_ecc_BSC(error_prob_values, n, k, m, num_bits):
    """
    Simulation using Reed-Solomon ECC with a BSC channel (no modulation).
    
    Parameters:
      - n, k, m: Reed-Solomon parameters (with block length m).
      - num_bits: Total number of bits to simulate.
      - error_prob_values: Iterable of BSC error probabilities.
    """
    ber_values = []
    for p in error_prob_values:
        # Generate random input bits.
        input_bits = np.random.randint(0, 2, size=num_bits, dtype=np.uint8)
        
        # Split the bits into blocks of length m.
        ecc_parser = BlockParser(block_size=m)
        parsed_blocks = ecc_parser.parse(input_bits)
        
        # RS ECC encoding.
        codec = ReedSolomonCodec(n=n, k=k, m=m)
        encoded_blocks = codec.encode(parsed_blocks)
        encoded_bits = ecc_parser.reconstruct(encoded_blocks)
        
        # Transmit over the BSC.
        channel = BinarySymmetricChannel(error_probability=p)
        received_bits = channel.transmit(encoded_bits)
        
        # RS ECC decoding.
        decoded_blocks = codec.decode(ecc_parser.parse(received_bits))
        reconstructed_bits = ecc_parser.reconstruct(decoded_blocks)
        
        # Calculate Bit Error Rate (BER).
        ber = 1 - np.sum(input_bits == reconstructed_bits) / len(input_bits) + 1e-15
        ber_values.append(ber)
    return ber_values


def simulate_ber_no_ecc_BSC(error_prob_values, num_bits):
    """
    Simulation without any ECC (raw transmission) over a BSC channel.
    
    Parameters:
      - num_bits: Total number of bits to simulate.
      - error_prob_values: Iterable of BSC error probabilities.
    """
    ber_values = []
    for p in error_prob_values:
        input_bits = np.random.randint(0, 2, size=num_bits, dtype=np.uint8)
        channel = BinarySymmetricChannel(error_probability=p)
        received_bits = channel.transmit(input_bits)
        ber = 1 - np.sum(input_bits == received_bits) / len(input_bits) + 1e-15
        ber_values.append(ber)
    return ber_values


def simulate_ber_with_polar_BSC(error_prob_values, N, K, num_bits):
    """
    Simulation using a polar code (with privacy amplification) over a BSC channel.
    
    In this simulation, we first pre-process the input bits with a privacy 
    amplification step (using PrivacyAmplification2), then encode with a polar code.
    After transmission over the BSC, the decoder recovers the privacy-amplified
    message, which is then de-hashed to recover the original bits.
    
    Parameters:
      - N: Polar code block length (must be a power of 2).
      - K: Number of information bits per polar block.
      - num_bits: Total number of bits to simulate.
      - error_prob_values: Iterable of BSC error probabilities.
    """
    ber_values = []
    for p in error_prob_values:
        input_bits = np.random.randint(0, 2, size=num_bits, dtype=np.uint8)
        
        # Create the polar codec and privacy amplification objects.
        polar_codec = PolarCodec(N=N, K=K)
        hash_obj = PrivacyAmplification2(256, 2048)
        
        # First, apply the privacy amplification encoder.
        pre_encoded_bits = hash_obj.encode(input_bits)
        # Then, apply the polar ECC encoder.
        encoded_bits = polar_codec.encode(pre_encoded_bits)
        
        # Transmit directly over the BSC.
        channel = BinarySymmetricChannel(error_probability=p)
        received_bits = channel.transmit(encoded_bits)
        
        # Decode with the polar code.
        decoded_bits = polar_codec.decode(received_bits)
        # And then de-hash.
        final_decoded_bits = hash_obj.decode(decoded_bits)
        
        ber = 1 - np.sum(input_bits == final_decoded_bits) / len(input_bits) + 1e-15
        ber_values.append(ber)
    return ber_values


# =============================================================================
# Main Simulation and Plotting
# =============================================================================
if __name__ == "__main__":
    # RS ECC parameters.
    n, k, m = 15, 9, 4   # (m must match the block length expected by the RS codec)
    
    # For the polar code simulation.
    N_polar = 32        # Polar code block length (power of 2)
    K_polar = 8         # Number of information bits per block (for polar ECC)
    
    num_bits = 1000000  # Total number of bits to simulate
    
    # Define a set of error probability values for the BSC.
    error_prob_values = np.linspace(0.001, 0.1, 10)
    
    # Run the simulations.
    ber_with_ecc   = simulate_ber_with_ecc_BSC(error_prob_values, n, k, m, num_bits)
    ber_no_ecc     = simulate_ber_no_ecc_BSC(error_prob_values, num_bits)
    ber_with_polar = simulate_ber_with_polar_BSC(error_prob_values, N_polar, K_polar, num_bits)
    
    # Plot the results.
    plt.figure()
    plt.semilogy(error_prob_values, ber_with_ecc, label=f"With RS ECC ({n},{k})", marker='o')
    plt.semilogy(error_prob_values, ber_with_polar, label=f"With Polar Code ({N_polar},{K_polar})", marker='s')
    plt.semilogy(error_prob_values, ber_no_ecc, label="Without ECC", marker='x')
    plt.title("BER vs Error Probability (BSC)")
    plt.xlabel("BSC Error Probability")
    plt.ylabel("Bit Error Rate (BER)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.show()

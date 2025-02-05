#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Import required modules (adjust import paths as needed)
from channels.baseChannels import ComplexAWGNChannel  # AWGN channel for both Bob and Eve
from modulation.baseModulation import MQUAMAntiGray,MQUAM     # Anti-Gray modulator
from codec.errorCodecs import PolarCodec                # Polar Code encoder/decoder
from util.parser import BlockParser                     # To parse bits for modulation

def simulate_wiretap_polar(snr_bob, snr_eve, N, K_list, M, num_bits):
    """
    Simulates a wiretap scenario where:
      - Alice encodes messages using a Polar code of block length N and various K (rate = K/N).
      - The encoded bits are modulated using Anti‑Gray MQUAM modulation.
      - Bob receives the modulated signal over a good channel (snr_bob in dB).
      - Eve receives the same signal over a degraded channel (snr_eve in dB).
    
    For each polar code rate (each value of K in K_list), the function computes
    the BER for Bob and for Eve.
    
    Args:
        snr_bob (float): SNR (in dB) of Bob's channel.
        snr_eve (float): SNR (in dB) of Eve's channel (degraded).
        N (int): Polar code block length (must be a power of 2).
        K_list (list[int]): List of information bit lengths (each gives rate = K/N).
        M (int): Modulation order for Anti-Gray MQUAM (e.g. 4 for QPSK).
        num_bits (int): Number of input bits to simulate.
    
    Returns:
        tuple: Two lists containing BER for Bob and BER for Eve, respectively.
    """
    ber_bob_list = []
    ber_eve_list = []
    
    for K in K_list:
        print(f"Simulating for Polar code: N={N}, K={K} (rate={K/N:.2f})")
        # Generate random input bits.
        input_bits = np.random.randint(0, 2, size=num_bits, dtype=np.uint8)
        
        # --- Polar Coding ---
        polar_codec = PolarCodec(N=N, K=K)
        encoded_bits = polar_codec.encode(input_bits)
        
        # --- Modulation (Anti-Gray MQUAM) ---
        # BlockParser will group bits in blocks of length M.
        mod_parser = BlockParser(block_size=M)
        modulation = MQUAM(M, power=1.0)
        modulated_signal = modulation.modulate(mod_parser.parse(encoded_bits))
        
        # --- Bob's Reception (Good Channel) ---
        channel_bob = ComplexAWGNChannel(snr_db=snr_bob)
        received_bob = channel_bob.transmit(modulated_signal)
        demodulated_bits_bob = mod_parser.reconstruct(modulation.demodulate(received_bob))
        decoded_bits_bob = polar_codec.decode(demodulated_bits_bob)
        
        # --- Eve's Reception (Degraded Channel) ---
        channel_eve = ComplexAWGNChannel(snr_db=snr_eve)
        received_eve = channel_eve.transmit(modulated_signal)
        demodulated_bits_eve = mod_parser.reconstruct(modulation.demodulate(received_eve))
        decoded_bits_eve = polar_codec.decode(demodulated_bits_eve)
        
        # --- BER Computation ---
        ber_bob = 1 - np.sum(input_bits == decoded_bits_bob) / len(input_bits) + 1e-15
        ber_eve = 1 - np.sum(input_bits == decoded_bits_eve) / len(input_bits) + 1e-15
        
        ber_bob_list.append(ber_bob)
        ber_eve_list.append(ber_eve)
        print(f"Rate: {K/N:.2f}, Bob BER: {ber_bob:.4e}, Eve BER: {ber_eve:.4e}")
    
    return ber_bob_list, ber_eve_list

if __name__ == "__main__":
    # Simulation parameters
    N_polar = 32               # Polar code block length (must be power of 2)
    # Vary K to adjust the Polar code rate (K/N)
    K_list = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    M = 16                      # Modulation order (4 for QPSK, here implemented with Anti-Gray MQUAM)
    num_bits = 4*10000          # Total number of input bits to simulate
    
    # Set channel SNRs (in dB)
    snr_bob = 7                # Bob's channel (good)
    snr_eve = 2                # Eve's channel (degraded)
    
    # Run the simulation over the range of polar code rates
    ber_bob, ber_eve = simulate_wiretap_polar(snr_bob, snr_eve, N_polar, K_list, M, num_bits)
    
    # Plot the BER curves vs. code rate (K/N)
    rates = [K / N_polar for K in K_list]
    plt.figure(figsize=(8, 6))
    plt.semilogy(rates, ber_bob, 'bo-', label="Bob BER")
    plt.semilogy(rates, ber_eve, 'rs-', label="Eve BER")
    plt.xlabel("Polar Code Rate (K/N)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("Wiretap Scenario: Bob vs Eve BER\n(Polar Codes + Anti-Gray MQUAM)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.show()

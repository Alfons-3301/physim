import numpy as np
import matplotlib.pyplot as plt
from channels.baseChannels import ComplexAWGNChannel
from modulation.baseModulation import MQAMModulation,MQUAM,QPSK
from codec.errorCodecs import ReedSolomonCodec, PolarCodec
from util.parser import BlockParser


def simulate_ber_with_ecc(snr_db_values, n, k, m, M, num_bits):
    ber_values = []
    
    for snr_db in snr_db_values:
        # Generate random bits
        input_bits = np.random.randint(0, 2, size=num_bits, dtype=np.uint8)

        # Parse bits into blocks
        ecc_parser = BlockParser(block_size=m)
        parsed_blocks = ecc_parser.parse(input_bits)

        # Reed-Solomon Codec
        codec = ReedSolomonCodec(n=n, k=k, m=m)
        encoded_blocks = codec.encode(parsed_blocks)

        # MQUAM Modulation
        mod_parser = BlockParser(block_size=M)
        modulation = QPSK()#MQUAM(M)
        modulated_signal = modulation.modulate(mod_parser.parse(encoded_blocks))

        # AWGN Channel
        channel = ComplexAWGNChannel(snr_db=snr_db)
        received_signal = channel.transmit(modulated_signal)

        # Demodulation
        demodulated_bits = mod_parser.reconstruct(modulation.demodulate(received_signal))

        # Decode with Reed-Solomon
        decoded_blocks = codec.decode(ecc_parser.parse(demodulated_bits))

        # Reconstruct the original bit sequence
        reconstructed_bits = ecc_parser.reconstruct(decoded_blocks)

        # Calculate BER
        ber = 1 - np.sum(input_bits == reconstructed_bits) / len(input_bits) + 1e-15
        ber_values.append(ber)

    return ber_values





def simulate_ber_no_ecc(snr_db_values, M, num_bits):
    ber_values = []

    for snr_db in snr_db_values:
        # Generate random bits
        input_bits = np.random.randint(0, 2, size=num_bits, dtype=np.uint8)

        # MQUAM Modulation
        mod_parser = BlockParser(block_size=M)
        modulation = QPSK()#MQUAM(M)
        modulated_signal = modulation.modulate(mod_parser.parse(input_bits))

        # AWGN Channel
        channel = ComplexAWGNChannel(snr_db=snr_db)
        received_signal = channel.transmit(modulated_signal)

        # Demodulation
        demodulated_bits = mod_parser.reconstruct(modulation.demodulate(received_signal))

        # Calculate BER
        ber = 1 - np.sum(input_bits == demodulated_bits) / len(input_bits) + 1e-15
        ber_values.append(ber)

    return ber_values

def simulate_ber_with_polar(snr_db_values, N, K, M, num_bits):
    """
    Simulate the BER of a system employing a simple polar code.
    
    The encoder/decoder is assumed to work with binary input (with automatic padding).
    For modulation, the encoded bits are mapped with QPSK.
    
    Note: The polar code implemented here uses exhaustive ML decoding,
    so it is only feasible for small values of K (e.g., K=8).
    """
    ber_values = []
    
    for snr_db in snr_db_values:
        # Generate random bits
        input_bits = np.random.randint(0, 2, size=num_bits, dtype=np.uint8)
        
        # Polar codec encoding
        polar_codec = PolarCodec(N=N, K=K)
        encoded_bits = polar_codec.encode(input_bits)
        
        # QPSK Modulation
        mod_parser = BlockParser(block_size=M)
        modulation = QPSK()
        modulated_signal = modulation.modulate(mod_parser.parse(encoded_bits))
        
        # AWGN Channel
        channel = ComplexAWGNChannel(snr_db=snr_db)
        received_signal = channel.transmit(modulated_signal)
        
        # Demodulation
        demodulated_bits = mod_parser.reconstruct(modulation.demodulate(received_signal))
        
        # Polar codec decoding
        decoded_bits = polar_codec.decode(demodulated_bits)
        
        # Calculate BER
        ber = 1 - np.sum(input_bits == decoded_bits) / len(input_bits) + 1e-15
        ber_values.append(ber)
    
    return ber_values


# Main Simulation
if __name__ == "__main__":
    # Parameters for RS ECC and uncoded system
    n, k, m = 15, 9, 4         # Reed-Solomon parameters (n, k, m)
    M = 2 ** 2                # QPSK modulation (effectively 4-QAM)
    num_bits = 1000000       # Total number of bits to simulate
    snr_db_values = np.linspace(2, 7, 10)  # Range of SNR values (in dB)
    
    # Parameters for the polar code (for demonstration, small parameters are chosen)
    N_polar = 32              # Block length (must be a power of 2)
    K_polar = 8               # Number of information bits per block
    
    # Simulate BER for each case
    ber_with_ecc = simulate_ber_with_ecc(snr_db_values, n, k, m, M, num_bits)
    ber_no_ecc = simulate_ber_no_ecc(snr_db_values, M, num_bits)
    ber_with_polar = simulate_ber_with_polar(snr_db_values, N_polar, K_polar, M, num_bits)
    
    # Plot results
    plt.figure()
    plt.semilogy(snr_db_values, ber_with_ecc, label=f"With RS ECC ({n},{k})", marker='o')
    plt.semilogy(snr_db_values, ber_with_polar, label=f"With Polar Code ({N_polar},{K_polar})", marker='s')
    plt.semilogy(snr_db_values, ber_no_ecc, label="Without ECC", marker='x')
    plt.title("BER vs SNR")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.show()
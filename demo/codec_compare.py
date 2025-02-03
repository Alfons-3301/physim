import numpy as np
import matplotlib.pyplot as plt
from channels.baseChannels import ComplexAWGNChannel
from modulation.baseModulation import MQAMModulation
from codec.errorCodecs import ReedSolomonCodec
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
        modulation = MQAMModulation(M)
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
        modulation = MQAMModulation(M)
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

# Main Simulation
if __name__ == "__main__":
    # Parameters
    n, k, m = 15, 9, 4  # Reed-Solomon (n, k, m)
    M = 2**4  # QAM order
    num_bits = 100000  # Number of bits
    snr_db_values = np.linspace(1, 20, 40)  # Range of SNR values

    # Simulate BER for both cases

    ber_with_ecc = simulate_ber_with_ecc(snr_db_values, n, k, m, M, num_bits)

    ber_no_ecc = simulate_ber_no_ecc(snr_db_values, M, num_bits)

    # Plot results
    plt.figure()

    plt.semilogy(snr_db_values, ber_with_ecc, label=f"With {n},{k} Reed-Solomon ECC", marker='o')
    plt.semilogy(snr_db_values, ber_no_ecc, label="Without ECC", marker='x')
    #plt.ylim((1e-15,1))
    plt.title("BER vs SNR")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.show()

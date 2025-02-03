from modulation.baseModulation import MQAMModulation
from channels.baseChannels import ComplexAWGNChannel, PhaseNoiseChannel
import numpy as np
import matplotlib.pyplot as plt


def simulate_awgn_mqam():
    M = 16  # 16-QAM
    snr_db = [1, 10, 15, 20, 40]  # SNR in dB
    power = 1.0  # Power constraint

    # Initialize Modulation and Channels
    mqam = MQAMModulation(M, power=power)
    channel = ComplexAWGNChannel(10)  # Placeholder SNR
    phase_noise = PhaseNoiseChannel(0*np.pi/16)

    # Generate random input bits
    num_bits = 4 * 4 * 4 * 4 * 4 * 7
    bits_per_symbol = int(np.log2(M))
    print(f"Bits per symbol: {bits_per_symbol}")

    for snr in snr_db:
        # Set channel SNR
        channel.set_parameters({"snr_db": snr})

        # Generate input bits and symbols
        input_bits = np.random.randint(0, 2, num_bits)
        input_symbols = mqam.bits_to_symbols(input_bits)

        # Modulate symbols and transmit
        modulated_signal = mqam.modulate(input_symbols)
        received_signal = phase_noise.transmit(channel.transmit(modulated_signal))

        # Demodulate symbols and recover bits
        demodulated_symbols = mqam.demodulate(received_signal)
        decoded_bits = mqam.symbols_to_bits(demodulated_symbols)

        # Compare original and decoded bits
        num_errors = np.sum(input_bits != decoded_bits[:num_bits])
        ber = num_errors / num_bits

        print(f"SNR: {snr} dB")
        print(f"Bit Errors: {num_errors}/{num_bits}")
        print(f"Bit Error Rate (BER): {ber:.4f}")

        # Visualization: Bits
        plt.figure(figsize=(10, 4))
        plt.plot(input_bits[:100], 'bo-', label='Original Bits', markersize=4)
        plt.plot(decoded_bits[:100], 'r.--', label='Decoded Bits', markersize=4)
        plt.title(f"Original vs Decoded Bits (SNR = {snr} dB, First 100 Bits)")
        plt.xlabel("Bit Index")
        plt.ylabel("Bit Value")
        plt.legend()
        plt.grid()
        plt.show()

        # Visualization: Constellation
        plt.figure(figsize=(8, 6))
        plt.scatter(modulated_signal.real, modulated_signal.imag, color='blue', label="Original", alpha=0.5)
        plt.scatter(received_signal.real, received_signal.imag, color='red', label="Received", alpha=0.5)
        plt.title(f"16-QAM Constellation with AWGN (SNR = {snr} dB)")
        plt.xlabel("In-Phase (I)")
        plt.ylabel("Quadrature (Q)")
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":
    simulate_awgn_mqam()

#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Import the two modulation classes.
# Adjust the imports as needed depending on your project structure.
from modulation.baseModulation import MQUAM,MQUAMAntiGray
from channels.baseChannels import ComplexAWGNChannel, PhaseNoiseChannel

def simulate_awgn_mqam_comparison():
    M = 4096  # 16-QAM
    snr_db = np.linspace(0.1,45,100) # SNR values in dB
    power = 1.0  # Power constraint

    # Use the same number of bits for both simulations.
    # Ensure that the number is a multiple of the bits-per-symbol (log2(M)).
    num_bits = 12 * 5000
    bits_per_symbol = int(np.log2(M))
    print(f"Bits per symbol: {bits_per_symbol}")

    # Instantiate the two modulation methods.
    normal_qam = MQUAM(M, power=power)
    anti_gray_qam = MQUAMAntiGray(M, power=power)

    # Create channel objects.
    channel = ComplexAWGNChannel(10)  # Placeholder SNR; will be updated
    phase_noise = PhaseNoiseChannel(0 * np.pi/16)

    # Prepare lists to store BER results.
    ber_normal = []
    ber_anti_gray = []

    for snr in snr_db:
        # Set channel SNR.
        channel.set_parameters({"snr_db": snr})

        # Generate random input bits.
        input_bits = np.random.randint(0, 2, num_bits)

        # --- Standard QAM Simulation ---
        modulated_normal = normal_qam.modulate(input_bits)
        # Pass through the channel (AWGN and phase noise).
        received_normal = phase_noise.transmit(channel.transmit(modulated_normal))
        # Demodulate and recover bits.
        decoded_normal = normal_qam.demodulate(received_normal)
        # Compute BER (compare only the first num_bits bits).
        errors_normal = np.sum(input_bits != decoded_normal[:num_bits])
        ber_n = errors_normal / num_bits
        ber_normal.append(ber_n)

        # --- Anti Gray QAM Simulation ---
        modulated_anti = anti_gray_qam.modulate(input_bits)
        received_anti = phase_noise.transmit(channel.transmit(modulated_anti))
        decoded_anti = anti_gray_qam.demodulate(received_anti)
        errors_anti = np.sum(input_bits != decoded_anti[:num_bits])
        ber_a = errors_anti / num_bits
        ber_anti_gray.append(ber_a)

        # Print the SNR and BER for both methods.
        print(f"SNR: {snr} dB")
        print(f"  Normal QAM BER:    {ber_n:.4f} ({errors_normal}/{num_bits} errors)")
        print(f"  Anti Gray QAM BER: {ber_a:.4f} ({errors_anti}/{num_bits} errors)")

    # Plot BER vs SNR for both modulation methods.
    plt.figure(figsize=(8, 6))
    plt.semilogy(snr_db, ber_normal, 'bo-', label='Normal QAM')
    plt.semilogy(snr_db, ber_anti_gray, 'rs-', label='Anti Gray QAM')
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("BER Comparison: Normal QAM vs Anti Gray QAM")
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.show()


if __name__ == "__main__":
    simulate_awgn_mqam_comparison()

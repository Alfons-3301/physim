from modulation.baseModulation import MQAMModulation
from channels.baseChannels import AWGNChannel
import matplotlib.pyplot as plt
import torch

# Simulation and Visualization

def simulate_awgn_mqam():
    M = 16  # 16-QAM
    snr_db = [1,10,15,20,40]  # SNR in dB
    power = 1.0  # Power constraint

    # Initialize Modulation and AWGN Channel
    mqam = MQAMModulation(M, power=power)
    channel = AWGNChannel(10)

    # Generate random input bits and symbols
    num_bits = 1000
    bits_per_symbol = int(torch.log2(torch.tensor(M,dtype=torch.float32)))
    print(f"Bits per symbol: {bits_per_symbol}")
    
    for snr in snr_db:
        channel.set_parameters({"snr_db":snr})
        input_bits = torch.randint(0, 2, (num_bits,))
        #print(f"input bits: {input_bits}")
        input_symbols = mqam.bits_to_symbols(input_bits)
        #print(f"input symbols: {input_symbols}")

        # Modulate symbols and transmit
        modulated_signal = mqam.modulate(input_symbols)
        received_signal = channel.transmit(modulated_signal)

        # Demodulate symbols and recover bits
        demodulated_symbols = mqam.demodulate(received_signal)
        decoded_bits = mqam.symbols_to_bits(demodulated_symbols)

        # Compare original and decoded bits
        num_errors = torch.sum(input_bits != decoded_bits[:len(input_bits)])
        ber = num_errors / num_bits

        print(f"Bit Errors: {num_errors}/{num_bits}")
        print(f"Bit Error Rate (BER): {ber:.4f}")

        # Visualization: Bits
        plt.figure(figsize=(10, 4))
        plt.plot(input_bits[:100], 'bo-', label='Original Bits', markersize=4)
        plt.plot(decoded_bits[:100], 'r.--', label='Decoded Bits', markersize=4)
        plt.title("Original vs Decoded Bits (First 100 Bits)")
        plt.xlabel("Bit Index")
        plt.ylabel("Bit Value")
        plt.legend()
        plt.grid()
        plt.show()

        # Visualization: Constellation
        plt.figure(figsize=(8, 6))
        plt.scatter(modulated_signal[:, 0], modulated_signal[:, 1], color='blue', label="Original", alpha=0.5)
        plt.scatter(received_signal[:, 0], received_signal[:, 1], color='red', label="Received", alpha=0.5)
        plt.title(f"16-QAM Constellation with AWGN (SNR = {snr} dB)")
        plt.xlabel("In-Phase (I)")
        plt.ylabel("Quadrature (Q)")
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == "__main__":
    simulate_awgn_mqam()
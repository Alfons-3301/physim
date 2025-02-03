import torch
import matplotlib.pyplot as plt
import numpy as np

# Import the channel classes
from channels.baseChannels import BinarySymmetricChannel, BAWGNChannel, AWGNChannel

# Helper function for visualization
def plot_histogram(data, title, xlabel, ylabel, bins=50, color='blue'):
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=bins, color=color, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()

# 1. Binary Symmetric Channel Visualization
def visualize_bsc():
    print("Visualizing Binary Symmetric Channel...")
    error_probability = 0.1
    channel = BinarySymmetricChannel(error_probability)

    # Generate random binary input
    input_bits = torch.randint(0, 2, (1000,)).float()
    output_bits = channel.transmit(input_bits)

    # Calculate flipped bits
    flipped_bits = (input_bits != output_bits).sum().item()
    print(f"Number of flipped bits: {flipped_bits} out of {len(input_bits)} (p = {error_probability})")

    # Visualize
    plt.figure(figsize=(8, 4))
    plt.scatter(range(len(input_bits)), input_bits, label="Input Bits", s=10, alpha=0.7)
    plt.scatter(range(len(output_bits)), output_bits, label="Output Bits (with Noise)", s=10, alpha=0.7)
    plt.title("Binary Symmetric Channel Behavior")
    plt.xlabel("Bit Index")
    plt.ylabel("Bit Value")
    plt.legend()
    plt.show()

# 2. BAWGN Channel Visualization
def visualize_bawgn():
    print("Visualizing Binary Additive White Gaussian Noise Channel...")
    snr_db = 5
    channel = BAWGNChannel(snr_db)

    # Generate binary input
    input_bits = torch.randint(0, 2, (1000,)).float()
    output_signal = channel.transmit(input_bits)

    # Visualize
    bpsk_signal = 2 * input_bits - 1  # BPSK mapping
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(bpsk_signal)), bpsk_signal, label="Original BPSK Signal", s=10, alpha=0.7)
    plt.scatter(range(len(output_signal)), output_signal, label="Received Signal (with Noise)", s=10, alpha=0.7)
    plt.title(f"BAWGN Channel Behavior (SNR = {snr_db} dB)")
    plt.xlabel("Signal Index")
    plt.ylabel("Signal Value")
    plt.legend()
    plt.grid()
    plt.show()

    # Histogram of received signal
    plot_histogram(output_signal.numpy(), 
                   title="Histogram of Received Signals in BAWGN Channel", 
                   xlabel="Received Signal Value", 
                   ylabel="Frequency", 
                   bins=50, color='green')

# 3. AWGN Channel Visualization
def visualize_awgn():
    print("Visualizing Additive White Gaussian Noise Channel...")
    snr_db = 10
    channel = AWGNChannel(snr_db)

    # Generate continuous input signal
    input_signal = torch.linspace(-1, 1, 1000)
    output_signal = channel.transmit(input_signal)

    # Visualize
    plt.figure(figsize=(8, 6))
    plt.plot(input_signal, label="Original Signal", alpha=0.7)
    plt.plot(output_signal, label="Received Signal (with Noise)", alpha=0.7)
    plt.title(f"AWGN Channel Behavior (SNR = {snr_db} dB)")
    plt.xlabel("Signal Index")
    plt.ylabel("Signal Value")
    plt.legend()
    plt.grid()
    plt.show()

    # Histogram of noise
    noise = output_signal - input_signal
    plot_histogram(noise.numpy(), 
                   title="Histogram of Noise in AWGN Channel", 
                   xlabel="Noise Value", 
                   ylabel="Frequency", 
                   bins=50, color='orange')

# Main Function to Run Visualizations
if __name__ == "__main__":
    visualize_bsc()
    visualize_bawgn()
    visualize_awgn()

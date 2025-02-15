import numpy as np
import matplotlib.pyplot as plt
from codec.protocolStack import ProtocolStack
from channels.baseChannels import BinarySymmetricChannel
from codec import errorCodecs
from codec.privacyAmp import PrivacyAmplification


def simulate_bsc_with_codec(bits, channel, codec):
    """
    simulate one packet, track BER, packet error
    """
    
    
    encoded_bits = codec.encode(bits)

    received_bits = channel.transmit(encoded_bits)

    decoded_bits = codec.decode(received_bits)

    ber = np.sum(bits != decoded_bits) / len(bits)
    packet_error = 1 if ber > 0 else 0

    return ber, packet_error


# Main simulation
if __name__ == "__main__":

    # Simulation parameters
    num_blocks = 50000
    n, k = 511, 40
    t = 95
    mu, fish = k//4, k  # mu: input bit vector length; fish: a parameter for PrivacyAmplification

    # Channel error probabilities to test
    p_values = 0.2*(np.linspace(0.4**(1/16), 1, 20) ** 16)
    # p_values = [0.0, 0.1, 0.2, 0.25]

    # Dictionary to store BER results for each Hash then Encode mode
    results = {}

    # Run simulation for both Hash then Encode modes: enabled and disabled
    for privacy_enabled in [True, False]:
        mode_name = "Hash then Encode" if privacy_enabled else "Only Encode"
        print("=" * 60)
        print(f"Running simulation with {mode_name}")
        
        # List to collect BER for each p value for this mode
        ber_list = []
        
        # Loop over each channel error probability
        for p in p_values:
            print("-" * 60)
            print(f"Channel error probability (p): {p}")
            
            # Instantiate the channel for current p
            channel = BinarySymmetricChannel(p)
            
            # Set up the protocol stack with PrivacyAmplification and an error-correcting codec
            codec = ProtocolStack(codecs=[
                PrivacyAmplification(k=mu, q=fish, enable=privacy_enabled),
                # You can add or swap in another codec if desired:
                errorCodecs.BCHCodec(data_block_size=k, code_block_size=n, correction_power=t),
            ])

            # Initialize error counters
            bit_errors = 0
            packet_errors = 0

            print(f"""
Simulation Parameters:
    Number of blocks:       {num_blocks}
    Bits per block (k):     {k}
    Total bits:             {num_blocks * k}
    Channel error probability (p): {p}
    Codec:                  {codec}
""")
            # Run simulation for the given number of blocks
            for i in range(num_blocks):
                # Generate random input bit vector of length 'mu'
                input_bits = np.random.randint(0, 2, size=mu, dtype=np.uint8)

                # Simulate the transmission of one packet through the channel with the codec
                ber, packet_error = simulate_bsc_with_codec(input_bits, channel, codec)

                bit_errors += ber
                packet_errors += packet_error

                # Optional: print progress every 1000 iterations
                if (i + 1) % 1000 == 0:
                    print(f"Progress: {i+1}/{num_blocks}", end="\r")

            # Compute final BER and PER for the current p value
            ber_final = bit_errors / num_blocks
            per_final = packet_errors / num_blocks

            # Append the BER for plotting later
            ber_list.append(ber_final)

            # Print out the results for this p value
            print("\n--- Results ---")
            print(f"p: {p}")
            print(f"Total packet errors: {packet_errors}")
            print(f"BER: {ber_final}")
            print(f"PER: {per_final}")
            if per_final > 0:
                print(f"Privacy amp performance (BER/PER): {ber_final / per_final}")
            else:
                print("Privacy amp performance (BER/PER): undefined (no packet errors)")
        
        # Save results for the current Hash then Encode mode
        results[mode_name] = np.array(ber_list)

    # Plot only the BER for both Hash then Encode modes versus channel error probability
    plt.figure(figsize=(10, 6))
    for mode_name, ber_list in results.items():
        ber_list[ber_list<(10**(-(np.log10(num_blocks * k))))] = 10**(-(np.log10(num_blocks * k)))
        plt.semilogy(p_values, ber_list, marker="o", label=mode_name,)
    plt.xlabel("Channel Error Probability (p)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("BER vs Channel Error Probability with/without Hash then Encode (BSC)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    for mode_name, ber_list in results.items():
        ber_list[ber_list<(10**(-(np.log10(num_blocks * k))))] = 10**(-(np.log10(num_blocks * k)))
        plt.plot(p_values, ber_list, marker="o", label=mode_name,)
    plt.xlabel("Channel Error Probability (p)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("BER vs Channel Error Probability with/without Hash then Encode (BSC)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
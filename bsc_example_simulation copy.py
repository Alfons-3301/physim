import numpy as np
import matplotlib.pyplot as plt
from codec.protocolStack import ProtocolStack
from channels.baseChannels import BinarySymmetricChannel
from codec.errorCodecs import PolarCodec_2,HammingCode
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


# Main Simulation
if __name__ == "__main__":

    num_blocks = 10000
    n, k = 4096, 1024
    µ, fish = 512, 1024
    p = 0.2

    channel = BinarySymmetricChannel(p)
    codec = ProtocolStack(codecs=[
        PrivacyAmplification(k=µ, q=fish),
        PolarCodec_2(N=n, K=k, design_p=0.1, decoder_type="SC", num_paths=8),
        # HammingCode(n=n, k=k),
    ])


    bit_errors = 0
    packet_errors = 0

    print(f"""
Simulation Parameters:
Number of blocks: {num_blocks}
Bits per block: {µ}
Total bits: {num_blocks * µ}
Channel error probability: {p}
Codec: {codec}
""")

    for i in range(num_blocks):
        # generate input bit vector
        input_bits = np.random.randint(0, 2, size=µ, dtype=np.uint8)

        # simulate one packet
        ber, packet_error = simulate_bsc_with_codec(input_bits, channel, codec)

        bit_errors += ber
        packet_errors += packet_error

        # \r print progress
        print(f"Progress: {i+1}/{num_blocks}", end="\r")

    ber = bit_errors / num_blocks
    bler = packet_errors / num_blocks

    print(f"Total packet errors: {packet_errors}")
    print(f"BER: {ber}")
    print(f"PER: {bler}")
    print(f"Privacy amp perf: {bler/ber}")



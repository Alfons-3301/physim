import numpy as np
import matplotlib.pyplot as plt
from channels.baseChannels import BinarySymmetricChannel
from codec.errorCodecs import PolarCodec_2
from codec.privacyAmp import PrivacyAmplification2
from util.parser import BlockParser


def simulate_bsc_with_polar(p_flip,µ,N, K, input_bits,codec):
    """
    Polar-coded simulation over a BSC
    """
    hash = PrivacyAmplification2(µ,K)
    
    hashed_bits = input_bits
    hashed_bits = hash.encode(input_bits)

    # Polar codec encoding
    encoded_bits = codec.encode(hashed_bits)
    
    # BSC
    channel = BinarySymmetricChannel(p_flip)
    received_bits = channel.transmit(encoded_bits)
    
    # Polar codec decoding
    decoded_bits = codec.decode(received_bits)

    decoded_bits = hash.decode(decoded_bits)
    # Calculate BER
    ber = 1.0 - np.sum(input_bits == decoded_bits) / len(input_bits) + 1e-15

    block_er = np.sum(np.any(input_bits.reshape((-1,µ)) != decoded_bits.reshape((-1,µ)),1)) / decoded_bits.reshape((-1,µ)).shape[0] + 1e-15

    return ber, block_er


# Main Simulation
if __name__ == "__main__":
    # Parameters for RS ECC and uncoded system
    num_bits = 1024*10        # Total number of bits to simulate
    input_bits = np.random.randint(0, 2, size=num_bits, dtype=np.uint8)

     # Parameters for the polar code (example: small rate, large N)
    N_polar = 4096
    K_polar = 1024

    R_hash = 1/2
    µ = int(K_polar*R_hash)
    
    codec_bob = PolarCodec_2(N=N_polar,K=K_polar,design_p=0.1,decoder_type="SCL",num_paths=8)
    codec_eve = PolarCodec_2(N=N_polar,K=K_polar,design_p=0.1,decoder_type="SCL",num_paths=32)

    ber_bob,bler_bob = simulate_bsc_with_polar(0.1,µ, N_polar, K_polar,input_bits,codec_bob)
    ber_eve,bler_eve = simulate_bsc_with_polar(0.2,µ, N_polar, K_polar,input_bits,codec_eve)
    print(f"BER@Bob:{ber_bob} BLER@Bob:{bler_bob} \nBER@Eve:{ber_eve} BLER@Eve:{bler_eve}")
    
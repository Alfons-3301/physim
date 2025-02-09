import numpy as np
import matplotlib.pyplot as plt
from channels.baseChannels import BinarySymmetricChannel
from codec.errorCodecs import PolarCodec_2,HammingCode
from codec.privacyAmp import PrivacyAmplification
from util.parser import BlockParser


def simulate_bsc_with_polar(p_flip,µ,hash, input_bits,codec):
    """
    Polar-coded simulation over a BSC
    """
    
    
    hashed_bits = input_bits
    #hashed_bits = hash.encode(input_bits)

    # Polar codec encoding
    encoded_bits = codec.encode(hashed_bits)
    
    # BSC
    channel = BinarySymmetricChannel(p_flip)
    received_bits = channel.transmit(encoded_bits)
    
    # Polar codec decoding
    decoded_bits = codec.decode(received_bits)

    # decoded_bits = hash.decode(decoded_bits)
    # Calculate BER
    ber = np.sum(input_bits != decoded_bits) / len(input_bits)

    # block_er = np.sum(np.all(input_bits.reshape((-1,µ)) != decoded_bits.reshape((-1,µ)),1)) / decoded_bits.reshape((-1,µ)).shape[0] + 1e-15
    block_er = int(ber > 0)
    return ber, block_er


# Main Simulation
if __name__ == "__main__":
    # Parameters for RS ECC and uncoded system
    K_polar = 1942
    num_bits = K_polar*100        # Total number of bits to simulate
    

     # Parameters for the polar code (example: small rate, large N)
    N_polar = 4096
    

    R_hash = 1/2
    µ = int(K_polar*R_hash)
    hash = PrivacyAmplification(µ,K_polar)
    
    codec_bob = PolarCodec_2(N=N_polar,K=K_polar,design_p=0.1,decoder_type="SCL",num_paths=256)
    #codec_eve = PolarCodec_2(N=N_polar,K=K_polar,design_p=0.1,decoder_type="SCL",num_paths=32)
    loops = 100
    ber_bob = 0
    bler_bob = 0
    ber_eve = 0
    bler_eve = 0
    for i in range(loops):
        input_bits = np.random.randint(0, 2, size=µ, dtype=np.uint8)
        ber_bob_1,bler_bob_1 = simulate_bsc_with_polar(0.1, µ, hash, input_bits, codec_bob)
        #ber_eve_1,bler_eve_1 = simulate_bsc_with_polar(0.2, µ, hash, input_bits, codec_eve)
        ber_bob += ber_bob_1/loops
        bler_bob += bler_bob_1/loops
        #ber_eve += ber_eve_1/loops
        #bler_eve += bler_eve_1/loops
        
    print(f"BER@Bob:{ber_bob} BLER@Bob:{bler_bob} \nBER@Eve:{ber_eve} BLER@Eve:{bler_eve}")
    
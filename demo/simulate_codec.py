import numpy as np
from channels.baseChannels import ComplexAWGNChannel
from modulation.baseModulation import MQAMModulation
from codec.errorCodecs import ReedSolomonCodec
from util.parser import BlockParser



# Pipeline Simulation
def pipeline_simulation():
    # Parameters
    n, k, m = 15, 8, 8  # Reed-Solomon (n, k) parameters
    M = 2**4 # QAM order
    snr_db = 10  # SNR for AWGN channel

    # Generate random bits
    num_bits = 1000
    input_bits = np.random.randint(0, 2, size=num_bits, dtype=np.uint8)

    # Parse bits into blocks
    ecc_parser = BlockParser(block_size=m)
    parsed_blocks = ecc_parser.parse(input_bits)

    # Reed-Solomon Codec
    codec = ReedSolomonCodec(n=n, k=k,m=m)
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

    # Verify correctness
    ber = 1 - np.sum(input_bits == reconstructed_bits) / len(input_bits) + 1e-15

    print("Pipeline Simulation Successful!")
    print(f"BER: {ber}")
    print(f"Input Bits: \t \t{input_bits[:20]}")
    print(f"Reconstructed Bits: \t{reconstructed_bits[:20]}")

# Run the simulation
if __name__ == "__main__":
    pipeline_simulation()

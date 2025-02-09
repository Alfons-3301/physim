import numpy as np
import matplotlib.pyplot as plt
import galois

# --- Helper functions for bit/integer conversions ---
def bits_to_int(bits):
    """Convert an array (or list) of bits (MSB first) to an integer."""
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result

def int_to_bits(x, length):
    """Convert an integer to a bit array of the specified length (MSB first)."""
    bits = [(x >> i) & 1 for i in range(length - 1, -1, -1)]
    return np.array(bits, dtype=int)

# --- Improved ei-UHF Implementation ---
class EiUHF:
    r"""
    Invertible Universal Hash Function (ei-UHF)
    -------------------------------------------
    
    Let 
      - \(X = \{0,1\}^r\) and 
      - \(Y = \{0,1\}^b\).
      
    The underlying multiplicative hash is defined as:
    \[
      h_s(x)=
      \begin{cases} 
         (s \cdot x)_{|b}, & \text{if } r \ge b, \\
         s \cdot (x\,\Vert\,0^{b-r}), & \text{if } r < b,
      \end{cases}
    \]
    where “\(\cdot\)” denotes multiplication in a finite field and
    \((\cdot)_{|b}\) extracts the first \(b\) bits of the field element’s
    representation.
    
    The invertible hash is then defined by
    \[
       g_s(x,y)= h_s(x) \oplus y.
    \]
    Its inversion is given by
    \[
       g_s^{\mathrm{Inv}}(t,x)= (x,\,h_s(x)\oplus t),
    \]
    so that if one transforms a message block \(m=(x,y)\) into
    \((x,t)\) with \(t=h_s(x)\oplus y\) and later recovers \(x\) and \(t\),
    one can retrieve the original \(y\) via \(y = h_s(x)\oplus t\).
    
    The finite field is chosen as GF(2^n) with
      \[
         n=\begin{cases}r, & r\ge b, \\[1mm] b, & r < b.\end{cases}
      \]
    """
    
    def __init__(self, r: int, b: int, s: np.ndarray = None):
        """
        Args:
            r (int): Length of x (bits).
            b (int): Length of y (and of the hash output t).
            s (np.ndarray, optional): Seed as a bit array. If r ≥ b, its length should be r; if r < b, its length should be b.
                If None, a random nonzero seed is generated.
        """
        self.r = r
        self.b = b
        self.n = r if r >= b else b
        self.GF = galois.GF(2**self.n)
        
        if s is None:
            s_int = np.random.randint(1, 2**self.n)
        else:
            if isinstance(s, np.ndarray):
                if len(s) != self.n:
                    raise ValueError(f"Seed must be a bit array of length {self.n}.")
                s_int = bits_to_int(s)
            else:
                s_int = int(s)
        self.s = self.GF(s_int)
    
    def h(self, x: np.ndarray) -> np.ndarray:
        """
        Computes h_s(x) as defined above.
        
        Args:
            x (np.ndarray): Bit array (length r).
        
        Returns:
            np.ndarray: Bit array (length b).
        """
        if len(x) != self.r:
            raise ValueError(f"Input x must be a bit array of length {self.r}.")
        x_int = bits_to_int(x)
        if self.r >= self.b:
            prod = self.s * self.GF(x_int)
            prod_int = int(prod)
            h_val = prod_int >> (self.r - self.b)
            return int_to_bits(h_val, self.b)
        else:
            padded = x_int << (self.b - self.r)
            prod = self.s * self.GF(padded)
            prod_int = int(prod)
            return int_to_bits(prod_int, self.b)
    
    def g(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes the forward transformation:
            g_s(x,y) = h_s(x) XOR y.
        
        Args:
            x (np.ndarray): Bit array of length r.
            y (np.ndarray): Bit array of length b.
        
        Returns:
            np.ndarray: Bit array of length b.
        """
        if len(y) != self.b:
            raise ValueError(f"Input y must be a bit array of length {self.b}.")
        hx = self.h(x)
        return np.bitwise_xor(hx, y)
    
    def invert(self, t: np.ndarray, x: np.ndarray):
        """
        Inverts the transformation. Given t = h_s(x) XOR y and x,
        recovers y.
        
        Args:
            t (np.ndarray): Bit array of length b.
            x (np.ndarray): Bit array of length r.
        
        Returns:
            Tuple: (x, y) with y = h_s(x) XOR t.
        """
        if len(t) != self.b:
            raise ValueError(f"t must be a bit array of length {self.b}.")
        hx = self.h(x)
        y = np.bitwise_xor(hx, t)
        return x, y

# =============================================================================
#  Supporting Modules (Assumed to be available)
# =============================================================================
from channels.baseChannels import ComplexAWGNChannel
from modulation.baseModulation import QPSK  # or MQUAM, etc.
from codec.errorCodecs import PolarCodec   # PolarCodec is our ECC here.
from util.parser import BlockParser

# =============================================================================
#  Simulation Pipeline: Polar Code ECC in a Hash-then-Encode System (HtE)
# =============================================================================
def simulate_ber_with_eihuf_polar(snr_db_values, N, K, M, num_bits, r, b):
    """
    Simulation using polar codes as the ECC in a hash-then-encode system.
    
    In this HtE system:
      - Each m-bit message block (with m = r + b) is split as (x, y),
        where x ∈ {0,1}^r and y ∈ {0,1}^b.
      - The transformation t = h(x) ⊕ y is computed using an ei-UHF.
      - The transformed block (x, t) is encoded using a polar code (with parameters N and K = m).
      - After transmission and decoding, the original y is recovered via y = h(x) ⊕ t.
    
    Args:
        snr_db_values (iterable): List or array of SNR values (in dB).
        N (int): Block length of the polar code (must be a power of 2).
        m (int): Message block length for the polar code (equals r+b).
        M (int): Block size for modulation.
        num_bits (int): Total number of bits to simulate.
        r (int): Number of bits allocated to the random seed x.
        b (int): Number of bits allocated to the message y.
                 (Thus, m must equal r + b.)
    
    Returns:
        List of BER values (one per SNR value).
    """
    if m != (r + b):
        raise ValueError("For ei-UHF processing, block size m must equal r + b.")
    
    ber_values = []
    # Instantiate the ei-UHF (seed is assumed known at the receiver).
    eiuhf = EiUHF(r, b)
    
    # Create a block parser to split the overall bit sequence into m-bit blocks.
    parser = BlockParser(block_size=m)
    
    for snr_db in snr_db_values:
        # Generate random bits.
        input_bits = np.random.randint(0, 2, size=num_bits, dtype=np.uint8)
        
        # Parse into m-bit blocks.
        parsed_blocks = parser.parse(input_bits)
        
        # Apply HtE: For each block, split into x (r bits) and y (b bits),
        # then compute t = h(x) ⊕ y, forming the transformed block (x, t).
        transformed_blocks = []
        for block in parsed_blocks:
            x = block[:r]
            y = block[r:]
            t = eiuhf.g(x, y)
            transformed_blocks.append(np.concatenate([x, t]))
        transformed_blocks = np.array(transformed_blocks)
        
        # Reconstruct the full bit sequence from the transformed blocks.
        transformed_bits = parser.reconstruct(transformed_blocks)
        
        # Polar ECC encoding.
        # Here we set the polar code information length K equal to m.
        polar_codec = PolarCodec(N=N, K=K)
        encoded_bits = polar_codec.encode(transformed_bits)
        
        # Modulation (using QPSK).
        mod_parser = BlockParser(block_size=M)
        modulation = QPSK()
        modulated_signal = modulation.modulate(mod_parser.parse(encoded_bits))
        
        # Transmit over AWGN channel.
        channel = ComplexAWGNChannel(snr_db=snr_db)
        received_signal = channel.transmit(modulated_signal)
        
        # Demodulation.
        demodulated_bits = mod_parser.reconstruct(modulation.demodulate(received_signal))
        
        # Polar ECC decoding.
        decoded_bits = polar_codec.decode(demodulated_bits)
        
        # Parse the decoded bits into m-bit blocks.
        decoded_blocks = parser.parse(decoded_bits)
        
        # Invert HtE: For each decoded block, split into x and t, then recover y = h(x) ⊕ t.
        recovered_blocks = []
        for block in decoded_blocks:
            x_decoded = block[:r]
            t_decoded = block[r:]
            _, y_decoded = eiuhf.invert(t_decoded, x_decoded)
            recovered_blocks.append(np.concatenate([x_decoded, y_decoded]))
        recovered_blocks = np.array(recovered_blocks)
        
        # Reconstruct the overall bit sequence.
        reconstructed_bits = parser.reconstruct(recovered_blocks)
        
        # Calculate the Bit Error Rate (BER).
        ber = 1 - np.sum(input_bits == reconstructed_bits) / len(input_bits) + 1e-15
        ber_values.append(ber)
    
    return ber_values

# =============================================================================
#  (Optional) Other Simulation Functions for Comparison
# =============================================================================
def simulate_ber_with_polar(snr_db_values, N, K, M, num_bits):
    """
    Simulation with a polar code (using exhaustive ML decoding) without HtE.
    """
    ber_values = []
    for snr_db in snr_db_values:
        input_bits = np.random.randint(0, 2, size=num_bits, dtype=np.uint8)
        polar_codec = PolarCodec(N=N, K=K)
        encoded_bits = polar_codec.encode(input_bits)
        mod_parser = BlockParser(block_size=M)
        modulation = QPSK()
        modulated_signal = modulation.modulate(mod_parser.parse(encoded_bits))
        channel = ComplexAWGNChannel(snr_db=snr_db)
        received_signal = channel.transmit(modulated_signal)
        demodulated_bits = mod_parser.reconstruct(modulation.demodulate(received_signal))
        decoded_bits = polar_codec.decode(demodulated_bits)
        ber = 1 - np.sum(input_bits == decoded_bits) / len(input_bits) + 1e-15
        ber_values.append(ber)
    return ber_values

# =============================================================================
#  Main Simulation and Plotting
# =============================================================================
if __name__ == "__main__":
    # Parameters for the HtE system using polar ECC:
    # Here m = r + b is the length of each message block before ECC.
    r, b = 8, 8
    m = r + b
    # For the polar code, choose N (block length, a power of 2) and set K = m.
    N_polar_eihuf = 32  # Polar code block length.
    # For the plain polar code simulation, we set K = number of information bits.
    K_polar = 8
    # Modulation parameter (e.g., QPSK uses 2 bits per symbol).
    M = 2**2  # For QPSK, M = 4.
    num_bits = 1000000
    snr_db_values = np.linspace(2, 7, 10)
    
    # Simulate various schemes:
    ber_plain_polar = simulate_ber_with_polar(snr_db_values, N=N_polar_eihuf, K=K_polar, M=M, num_bits=num_bits)
    ber_eihuf_polar = simulate_ber_with_eihuf_polar(snr_db_values, N=N_polar_eihuf, K=K_polar, M=M, num_bits=num_bits, r=r, b=b)
    
    # Plot the results.
    plt.figure()
    plt.semilogy(snr_db_values, ber_plain_polar, label=f"Polar Code (N={N_polar_eihuf},K={K_polar})", marker='s')
    plt.semilogy(snr_db_values, ber_eihuf_polar, label=f"Polar ECC + HtE (r={r},b={b})", marker='^')
    plt.title("BER vs SNR (Polar ECC based HtE vs. Plain Polar ECC)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.show()

import numpy as np
import galois

from codec.baseClasses import AbstractCodec
from util.bit_padder import BitPadder
from util.parser import ReedSolomonAdapter

class HammingCode(AbstractCodec):
    """
    A Hamming-code encoder/decoder that:
      - Operates over a specified Galois Field (gf).
      - Accepts 1D input bit arrays of arbitrary length.
      - Automatically pads/unpads to fit the (n,k) block constraints.
    """

    def __init__(self, n: int, k: int, gf: galois.GF = galois.GF2):
        """
        Initializes the Reed-Solomon codec.

        Args:
            n (int): Codeword length.
            k (int): Message length.
            gf (galois.GF): The Galois Field to use (default galois.GF2).
        """
        self.n = n             # Codeword length
        self.k = k             # Message length
        self.GF = gf
        self.padder = BitPadder()

        # Generate generator (G) and parity-check (H) matrices in GF
        self._generate_matrices()

    def _generate_matrices(self):
        """
        Generates the generator matrix (G) and parity-check matrix (H).
        The actual generation logic can be adjusted to suit your Hamming variant.
        """
        # ---------------------------------------------------------------
        # Minimal demonstration approach:
        # For a (7,4) Hamming code, a standard G in GF2 looks like:
        #
        #     G = [
        #       [1, 0, 0, 0, 0, 1, 1],
        #       [0, 1, 0, 0, 1, 0, 1],
        #       [0, 0, 1, 0, 1, 1, 0],
        #       [0, 0, 0, 1, 1, 1, 1],
        #     ]
        #
        # This is just an example. For other n,k, adapt accordingly.
        #
        # We'll do a naive approach:
        #  - G = [I_k | P],  where P is an array of shape (k, n-k)
        #  - H = [P^T | I_{n-k}]
        # This is not always "canonical" Hamming, but good for demonstration.
        # ---------------------------------------------------------------

        # Identity portion: k x k
        I_k = np.eye(self.k, dtype=int)
        # We make a random (k x (n-k)) parity portion, ensuring it is in GF
        P = np.random.randint(0, 2, size=(self.k, self.n - self.k))

        # Generator matrix: G = [I_k | P]
        G_np = np.hstack((I_k, P))

        # Parity check matrix: H = [P^T | I_{n-k}]
        # shape: (n-k) x n
        I_r = np.eye(self.n - self.k, dtype=int)
        H_np = np.hstack((P.T, I_r))

        # Convert to GF
        self.G = self.GF(G_np)
        self.H = self.GF(H_np)

    def encode(self, input_bits: np.ndarray) -> np.ndarray:
        """
        Encodes input bits using the Hamming code.
        Automatically pads the input if not multiple of k.

        Args:
            input_bits (np.ndarray): 1D array of bits to encode.

        Returns:
            np.ndarray: 1D array of encoded bits (multiple codewords).
        """
        # 1) Pad input to make it multiple of k
        padded_bits, self._pad_len = BitPadder.pad(input_bits, self.k)


        # 2) Reshape to blocks of size k
        block_count = len(padded_bits) // self.k
        reshaped = padded_bits.reshape(block_count, self.k)

        # 3) Encode each block in GF
        reshaped_gf = self.GF(reshaped)
        codewords = reshaped_gf @ self.G  # (block_count, n)
        codewords = np.array(codewords)

        # 4) Flatten the codewords into a 1D array of bits
        encoded_bits = codewords.view(np.ndarray).astype(int).flatten()
        return encoded_bits

    def decode(self, received_signal: np.ndarray) -> np.ndarray:
        """
        Decodes the received signal using the Hamming code.
        Removes any zero-bit padding added during encode().

        Args:
            received_signal (np.ndarray): 1D array of received bits (must be multiple of n).

        Returns:
            np.ndarray: 1D array of recovered message bits (with padding removed).
        """
        # 1) Reshape the incoming bits to blocks of n
        if len(received_signal) % self.n != 0:
            raise ValueError(
                f"Length of received_signal ({len(received_signal)}) must be multiple of codeword length n={self.n}."
            )
        block_count = len(received_signal) // self.n
        received_blocks = received_signal.reshape(block_count, self.n)
        received_gf = self.GF(received_blocks)

        # 2) Syndrome
        syndromes = received_gf @ self.H.T  # shape: (block_count, n-k)

        # 3) Correct single-bit errors if the syndrome is non-zero
        corrected = received_gf.copy()
        for i in range(block_count):
            s = syndromes[i]
            # Convert GF row -> integer index (binary)
            s_int = int("".join(map(str, s.view(np.ndarray).astype(int))), 2)
            # For a standard Hamming code, the syndrome can be used to identify bit positions
            # We do an offset check; if s_int is non-zero, it indicates the bit position that is flipped.
            # This naive approach might differ from a canonical approach for some Hamming parameter sets.
            # Typically s_int in [1..n].
            if s_int != 0 and 1 <= s_int <= self.n:
                # Flip that bit in corrected codeword
                bit_pos = s_int - 1  # zero-based index
                corrected[i, bit_pos] ^= 1

        # 4) Extract the original message bits from codewords
        #    (we assume the first k bits are the message portion if G = [I_k | P]).
        recovered_bits = corrected[:, :self.k].view(np.ndarray).astype(int).flatten()

        # 5) Unpad to remove the zero bits added during encoding
        recovered_bits = self.padder.unpad(recovered_bits, self._pad_len)
        return recovered_bits
    

class ReedSolomonCodec(AbstractCodec):
    def __init__(self, n: int, k: int, m : int):
        """
        Initializes the Reed-Solomon codec.

        Args:
            n (int): Codeword length.
            k (int): Message length.
        """
        self.n = n
        self.k = k
        self.m = m
        self.parser = ReedSolomonAdapter(k)
        self.field = galois.GF(2**m)  # Working in GF(256)
        self.rs = galois.ReedSolomon(n, k, field=self.field)

    def _bits_to_symbols(self, bits: np.ndarray) -> np.ndarray:
        """
        Converts a bit sequence into integers.

        Args:
            bits (np.ndarray): Input bit sequence (1D array).

        Returns:
            np.ndarray: Array of integers (1D array).
        """
        if len(bits[1]) % self.m != 0:
            raise ValueError(f"Bit sequence length must be a multiple of {self.m}.")
        symbols = np.array([int("".join(map(str, block)), 2) for block in bits])
        return symbols
    


    def _symbols_to_bits(self, symbols: np.ndarray) -> np.ndarray:
        """
        Converts symbols in GF(2^m) back to bit blocks of length m.
        
        Parameters:
            symbols (array-like): Array of symbols (integers in GF(2^m)).
            m (int): Bit length of each block (field size is GF(2^m)).
        
        Returns:
            np.ndarray: A 2D numpy array of shape [B, m], where B is the number of blocks.
        """
        # Convert each symbol into its binary representation, padded to m bits
        bit_blocks = np.array([list(map(int, f"{symbol:0{self.m}b}")) for symbol in symbols])
        return bit_blocks

    def encode(self, input_bits: np.ndarray) -> np.ndarray:
        """
        Encodes the input bits into a codeword.

        Args:
            input_bits (np.ndarray): Input bit sequence as a 1D array.

        Returns:
            np.ndarray: Encoded codeword as a 1D bit array.
        """
        # Convert bits to integers
        input_integers = self._bits_to_symbols(input_bits)
        # Convert integers to GF elements
        #input_gf = self.field(input_integers)
        # Apply Reed-Solomon encoding
        adapted_symbols = self.parser.parse(input_integers)
        encoded_gf = self.rs.encode(adapted_symbols)
        # Convert GF elements back to integers
        encoded_integers = np.array(encoded_gf)
        # Convert integers back to bits
        return self._symbols_to_bits(encoded_integers.flatten()).flatten()

    def decode(self, received_signal: np.ndarray) -> np.ndarray:
        """
        Decodes the received signal back into the original input bits.

        Args:
            received_signal (np.ndarray): Received codeword as a 1D bit array.

        Returns:
            np.ndarray: Decoded message as a 1D bit array.
        """
        # Convert bits to integers
        received_integers = self._bits_to_symbols(received_signal)
        # Apply Reed-Solomon decoding
        adapted_symbols = received_integers.reshape(-1,self.n)
        decoded_gf = self.rs.decode(adapted_symbols)
        decoded_gf = self.parser.reconstruct(decoded_gf)
        # Convert GF elements back to integers
        decoded_integers = np.array(decoded_gf)
        # Convert integers back to bits
        return self._symbols_to_bits(decoded_integers)



class PolarCodec(AbstractCodec):
    """
    A simple Polar Codec that:
      - Uses a full polar transform with no frozen bits.
      - Encodes via F^{\otimes m}.
      - Decodes via a basic Successive Cancellation (SC) approach using LLRs.
    """

    def __init__(self, N: int):
        """
        Initialize with code length N (must be a power of 2 for standard polar).
        """
        self.N = N
        # In a more realistic setting, we would define which indices are frozen.
        # For demonstration, we freeze none:
        self.frozen_bits = np.array([False]*N)  # All are data bits.

    def encode(self, input_bits: np.ndarray) -> np.ndarray:
        """
        Encode the input bits into a polar codeword using the polar transform.
        input_bits: np.ndarray of shape (N,) with binary values {0,1}
        returns: codeword of shape (N,) also with binary {0,1}
        """
        assert len(input_bits) == self.N, "Input length must match N."
        # Convert to int type for safe XOR handling
        u = input_bits.astype(int)
        x = self._polar_transform(u)
        return x

    def decode(self, received_signal: np.ndarray) -> np.ndarray:
        """
        Decode the received signal back into the original input bits using
        Successive Cancellation (SC) decoding.

        received_signal: np.ndarray of shape (N,) - real-valued or LLRs
        returns: estimated input bits, shape (N,)
        """
        assert len(received_signal) == self.N, "Signal length must match N."
        # If these are raw BPSK symbols, for example, we can convert to LLRs.
        # For demonstration, let's assume a simple approximate LLR = +large if y>0, -large if y<0
        llrs = np.array([10.0 if y >= 0 else -10.0 for y in received_signal], dtype=float)

        # Now apply SC decoding
        u_hat = self._sc_decode(llrs, self.frozen_bits)
        return u_hat

    def _polar_transform(self, u: np.ndarray) -> np.ndarray:
        """
        Recursively apply the polar transform to array u of length N.
        """
        N = len(u)
        if N == 1:
            return u
        else:
            # Split in half, transform each recursively
            mid = N // 2
            u_upper = self._polar_transform(u[:mid])
            u_lower = self._polar_transform(u[mid:])
            # Combine (in GF(2), addition is XOR)
            x_upper = u_upper ^ u_lower
            x_lower = u_lower
            return np.concatenate([x_upper, x_lower])

    def _sc_decode(self, llrs: np.ndarray, frozen_bits: np.ndarray) -> np.ndarray:
        """
        Recursively perform Successive Cancellation decoding.
        llrs: Log-likelihood ratios for each position in a sub-block.
        frozen_bits: Boolean array telling which positions are frozen.

        Returns the estimated bits (0/1).
        """
        N = len(llrs)
        if N == 1:
            # If this bit were frozen, we would force it to 0 (typical).
            # Here, we assume not frozen. Decide 0 if LLR>0, else 1
            bit_est = 0 if llrs[0] >= 0 else 1
            return np.array([bit_est], dtype=int)

        # Split
        half = N // 2
        # f-function combines "upper" and "lower" LLR sets
        L_upper = self._f_function(llrs[:half], llrs[half:])
        # Recursively decode the upper half
        u_upper_hat = self._sc_decode(L_upper, frozen_bits[:half])

        # g-function uses the known upper bits to refine the lower's LLR
        L_lower = self._g_function(llrs[:half], llrs[half:], u_upper_hat)
        # Recursively decode the lower half
        u_lower_hat = self._sc_decode(L_lower, frozen_bits[half:])

        # Combine the estimates (the final estimate of the codeword bits 
        # in the standard SC approach is [u_upper_hat, u_lower_hat], 
        # but these are the *information* domain bits).
        return np.concatenate([u_upper_hat, u_lower_hat])

    @staticmethod
    def _f_function(llr_a: np.ndarray, llr_b: np.ndarray) -> np.ndarray:
        """
        f-function in SC decoding (approx. min-sum):
        f(a, b) = sign(a)*sign(b)*min(|a|, |b|)
        """
        return np.sign(llr_a) * np.sign(llr_b) * np.minimum(np.abs(llr_a), np.abs(llr_b))

    @staticmethod
    def _g_function(llr_a: np.ndarray, llr_b: np.ndarray, u_upper_hat: np.ndarray) -> np.ndarray:
        """
        g-function in SC decoding:
        g(a, b, u) = b + (-1)^{u} * a
        where u is the hard-decision bit for the upper half.
        If u=0, no sign flip; if u=1, flips sign of a.
        """
        # Convert bits {0,1} into {+1,-1} for sign flipping
        sign_flip = 1 - 2*u_upper_hat  # u=0 -> +1, u=1 -> -1
        return llr_b + sign_flip * llr_a
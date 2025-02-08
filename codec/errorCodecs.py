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

# (Assume BitPadder is defined elsewhere; if not, a simple implementation is provided below.)
class BitPadder:
    @staticmethod
    def pad(bits: np.ndarray, block_size: int):
        pad_len = (-len(bits)) % block_size
        if pad_len:
            padded = np.concatenate([bits, np.zeros(pad_len, dtype=int)])
        else:
            padded = bits.copy()
        return padded, pad_len

    @staticmethod
    def unpad(bits: np.ndarray, pad_len: int):
        if pad_len:
            return bits[:-pad_len]
        else:
            return bits

# -----------------------------------------------------------------------------
# PolarCode: A simple polar code for the BSC using exhaustive ML decoding.
# -----------------------------------------------------------------------------
class PolarCodec(AbstractCodec):
    """
    A simple polar-code encoder/decoder for the binary symmetric channel (BSC).

    The encoder accepts 1D input bit arrays (of arbitrary length) and
    automatically pads to fill blocks of K message bits. Encoding is done
    using the polar transform F^{\otimes n} (with F = [[1,0],[1,1]]) and by placing
    the K information bits in the positions determined by a reliability rule.
    Frozen bits (positions not in the information set) are set to zero.

    The decoder performs maximum-likelihood (ML) decoding by exhaustively searching
    over the 2^K possible messages. (This brute-force approach is only practical
    for small values of K.)
    """

    def __init__(self, N: int, K: int, design_p: float = 0.11):
        """
        Initializes the polar-code encoder/decoder.

        Args:
            N (int): Block length (must be a power of 2).
            K (int): Number of information bits per block.
            design_p (float): Design BSC error probability (used in selecting the
                              information set). (Not used in full detail here.)
        """
        if not (N & (N - 1) == 0 and N != 0):
            raise ValueError("N must be a power of 2.")
        if K > N:
            raise ValueError("K must be less than or equal to N.")

        self.N = N
        self.K = K
        self.design_p = design_p  # design parameter (for a real polar code, one would compute reliabilities)
        self.padder = BitPadder()

        # Compute the full polar transformation matrix F^{⊗n} (over GF(2))
        self.F = self._compute_polar_matrix(self.N)

        # Determine the information set using a simple heuristic.
        # Here we choose the K rows of F with the highest weight
        # (i.e. rows with the most ones). For a polar code on the BSC,
        # higher row-weight tends to indicate a more reliable bit-channel.
        row_weights = [(i, np.sum(self.F[i, :])) for i in range(self.N)]
        # Sort in descending order of weight; break ties with the index
        sorted_rows = sorted(row_weights, key=lambda x: (x[1], x[0]), reverse=True)
        # Select the indices corresponding to the K most reliable bit-channels
        info_indices = sorted([i for i, _ in sorted_rows[:self.K]])
        self.info_set = info_indices
        self.frozen_set = [i for i in range(self.N) if i not in self.info_set]

    def _compute_polar_matrix(self, N: int) -> np.ndarray:
        """
        Computes the polar transformation matrix F^{⊗n} for block length N.

        Args:
            N (int): Block length (must be a power of 2).

        Returns:
            np.ndarray: An (N x N) binary matrix.
        """
        F = np.array([[1, 0], [1, 1]], dtype=int)
        n = int(np.log2(N))
        F_n = F.copy()
        for _ in range(1, n):
            F_n = np.kron(F_n, F)
        # All operations are in GF(2); we use modulo 2 arithmetic.
        return F_n % 2

    def encode(self, input_bits: np.ndarray) -> np.ndarray:
        """
        Encodes input bits using the polar code.
        Automatically pads the input if it is not a multiple of K.

        Args:
            input_bits (np.ndarray): 1D array of bits to encode.

        Returns:
            np.ndarray: 1D array of encoded bits (concatenation of codewords of length N).
        """
        # 1) Pad input to make its length a multiple of K.
        padded_bits, self._pad_len = self.padder.pad(input_bits, self.K)

        # 2) Reshape into blocks of K bits.
        block_count = len(padded_bits) // self.K
        messages = padded_bits.reshape(block_count, self.K)

        codewords = []
        for m in messages:
            # Construct the full N-length input vector u with frozen bits = 0.
            u = np.zeros(self.N, dtype=int)
            # Place the message bits into the positions of the information set.
            u[self.info_set] = m
            # Compute the codeword: x = u * F^{⊗n} (mod 2).
            x = np.mod(np.dot(u, self.F), 2)
            codewords.append(x)
        encoded_bits = np.concatenate(codewords)
        return encoded_bits

    def decode(self, received_bits: np.ndarray) -> np.ndarray:
        """
        Decodes the received bits using exhaustive maximum-likelihood (ML) decoding.
        (This simple decoder is feasible only for small values of K.)

        Args:
            received_bits (np.ndarray): 1D array of received bits (must be a multiple of N).

        Returns:
            np.ndarray: 1D array of recovered message bits (with padding removed).
        """
        if len(received_bits) % self.N != 0:
            raise ValueError("Length of received_bits must be a multiple of N.")
        block_count = len(received_bits) // self.N
        received_blocks = received_bits.reshape(block_count, self.N)
        decoded_messages = []

        # Precompute all candidate messages (as binary arrays of length K).
        num_candidates = 2 ** self.K
        candidate_msgs = np.array([list(np.binary_repr(i, width=self.K))
                                   for i in range(num_candidates)], dtype=int)

        # For each candidate message, precompute the corresponding codeword.
        candidate_codewords = []
        for m in candidate_msgs:
            u = np.zeros(self.N, dtype=int)
            u[self.info_set] = m
            x = np.mod(np.dot(u, self.F), 2)
            candidate_codewords.append(x)
        candidate_codewords = np.array(candidate_codewords)

        # For each received block, find the candidate codeword that minimizes the Hamming distance.
        for y in received_blocks:
            # Compute Hamming distances (number of differing bits).
            distances = np.sum(np.abs(candidate_codewords - y), axis=1)
            best_idx = np.argmin(distances)
            best_message = candidate_msgs[best_idx]
            decoded_messages.append(best_message)
        decoded_bits = np.concatenate(decoded_messages)
        # Remove padding that was added during encoding.
        decoded_bits = self.padder.unpad(decoded_bits, self._pad_len)
        return decoded_bits

class PolarCodec_ingo(AbstractCodec):
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
    
class PolarCodec_2(AbstractCodec):
    """
    A simple polar-code encoder/decoder for the binary symmetric channel (BSC).

    This class serves as a wrapper around the hermespy AFF3CT-based Polar
    implementations. It follows the style of the example `PolarCodec(AbstractCodec)`.
    
    The encoder accepts 1D input bit arrays (of arbitrary length) and automatically
    pads to fill blocks of K message bits. Encoding is done via the underlying
    AFF3CT-based Polar SC or Polar SCL coder.

    The decoder uses the SC or SCL algorithm (depending on initialization) to
    recover the original data bits.

    Assumes that the padding length is always known (and stored here) between encode and decode.
    """

    def __init__(
        self,
        N: int,
        K: int,
        design_p: float = 0.11,
        decoder_type: str = "SCL",
        num_paths: int = 8
    ):
        """
        Initializes the polar-code encoder/decoder using the hermespy AFF3CT-based
        Polar classes.

        Args:
            N (int): Code block size (number of code bits per encoded block).
            K (int): Data block size (number of data bits per block).
            design_p (float): Assumed BSC error probability (used internally by AFF3CT).
            decoder_type (str): 'SC' for successive cancellation or 'SCL' for
                                successive cancellation list decoding.
            num_paths (int): Number of decoding paths (only relevant if decoder_type='SCL').
        """
        if K > N:
            raise ValueError("K must be less than or equal to N.")

        self.N = N
        self.K = K
        self.design_p = design_p  # an approximate or design BER
        self._padder = BitPadder()

        # We will store the last pad length used in encode so that decode can unpad properly.
        self._pad_len = 0

        # Instantiate the underlying AFF3CT-based Polar coder
        if decoder_type.upper() == "SC":
            # hermespy Polar Successive Cancellation
            # constructor signature: PolarSCCoding(data_block_size, code_block_size, ber)
            from hermespy.fec.aff3ct.polar import PolarSCCoding
            self._polar_coder = PolarSCCoding(self.K, self.N, self.design_p)
        elif decoder_type.upper() == "SCL":
            # hermespy Polar Successive Cancellation List
            # constructor signature: PolarSCLCoding(data_block_size, code_block_size, ber, num_paths)
            from hermespy.fec.aff3ct.polar import PolarSCLCoding
            self._polar_coder = PolarSCLCoding(self.K, self.N, self.design_p, num_paths)
        else:
            raise ValueError("Unsupported decoder_type. Use 'SC' or 'SCL'.")

    def encode(self, input_bits: np.ndarray) -> np.ndarray:
        """
        Encodes input bits using the hermespy AFF3CT-based Polar coder.
        Automatically pads the input if it is not a multiple of K.

        Args:
            input_bits (np.ndarray): 1D array of bits to encode.

        Returns:
            np.ndarray: 1D array of encoded bits (concatenation of codewords of length N).
        """
        # 1) Pad input to make its length a multiple of K.
        padded_bits, pad_len = self._padder.pad(input_bits, self.K)
        self._pad_len = pad_len  # Store pad length for later removal in decode.

        # 2) Reshape into blocks of K bits.
        block_count = len(padded_bits) // self.K
        data_blocks = padded_bits.reshape(block_count, self.K)

        # 3) Encode each block using the AFF3CT-based coder.
        encoded_blocks = []
        for block in data_blocks:
            # The AFF3CT coder expects an np.ndarray of int32 for encode/decode
            block_i32 = block.astype(np.int32)
            encoded_block = self._polar_coder.encode(block_i32)
            encoded_blocks.append(encoded_block)

        # 4) Concatenate the codewords into a 1D array.
        encoded_bits = np.concatenate(encoded_blocks)
        return encoded_bits

    def decode(self, received_bits: np.ndarray) -> np.ndarray:
        """
        Decodes the received bits using the hermespy AFF3CT-based Polar coder.

        Args:
            received_bits (np.ndarray): 1D array of received bits (must be a multiple of N).

        Returns:
            np.ndarray: 1D array of recovered message bits (with padding removed).
        """
        if len(received_bits) % self.N != 0:
            raise ValueError("Length of received_bits must be a multiple of N.")

        block_count = len(received_bits) // self.N
        received_blocks = received_bits.reshape(block_count, self.N)

        # 1) Decode each block using the AFF3CT-based coder.
        decoded_blocks = []
        for block in received_blocks:
            block_i32 = block.astype(np.int32)
            decoded_block = self._polar_coder.decode(block_i32)
            decoded_blocks.append(decoded_block)

        # 2) Concatenate decoded bits into a single array.
        decoded_bits_all = np.concatenate(decoded_blocks)

        # 3) Remove the padding that was added during encode, 
        #    using the known pad length (self._pad_len).
        decoded_bits = self._padder.unpad(decoded_bits_all, self._pad_len)
        return decoded_bits

    @property
    def bit_block_size(self) -> int:
        """
        Number of bits within a data block to be encoded.
        (Equivalent to K.)
        """
        return self._polar_coder.bit_block_size

    @property
    def code_block_size(self) -> int:
        """
        Number of bits within a code block to be decoded.
        (Equivalent to N.)
        """
        return self._polar_coder.code_block_size

    @property
    def enabled(self) -> bool:
        """
        Whether this codec is enabled for encoding/decoding.
        """
        return self._polar_coder.enabled

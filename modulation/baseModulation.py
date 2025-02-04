import numpy as np
from modulation.baseClasses import AbstractModulation


class BPSKModulation(AbstractModulation):
    def modulate(self, input_data: np.ndarray) -> np.ndarray:
        """
        Modulates binary input {0, 1} to BPSK symbols {-1, +1}.
        """
        return 2 * input_data - 1.0  # Map: 0 -> -1, 1 -> +1

    def demodulate(self, received_signal: np.ndarray) -> np.ndarray:
        """
        Demodulates BPSK symbols back to binary {0, 1}.
        """
        return (received_signal > 0).astype(int)  # Threshold at 0

class QPSK(AbstractModulation):
    def __init__(self, power: float = 1.0):
        """
        Initializes QPSK modulation with Gray encoding.

        Args:
            power (float): Desired average power of the constellation.
        """
        self.M = 4
        self.bits_per_symbol = 2
        self.power = power
        self.constellation = self._generate_constellation()

    def _generate_constellation(self) -> np.ndarray:
        """
        Generates the QPSK constellation points with Gray mapping.

        The mapping is defined as follows (before normalization):
            Symbol index 0 (Gray code 00):  1 + 1j
            Symbol index 1 (Gray code 01): -1 + 1j
            Symbol index 2 (Gray code 11): -1 - 1j
            Symbol index 3 (Gray code 10):  1 - 1j

        Since the average energy of the base constellation is 2,
        a normalization factor of sqrt(power / 2) is applied.

        Returns:
            np.ndarray: Array of normalized complex constellation points.
        """
        base_constellation = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=complex)
        normalization_factor = np.sqrt(self.power / 2)
        return base_constellation * normalization_factor

    def modulate(self, input_bits: np.ndarray) -> np.ndarray:
        """
        Maps input bits to QPSK constellation points.

        Args:
            input_bits (np.ndarray): A 1D binary bitstream. Its length must be a multiple of 2.
        Returns:
            np.ndarray: Complex-valued constellation points.
        """
        symbols = self.bits_to_symbols(input_bits)
        return self.constellation[symbols]

    def demodulate(self, received_signal: np.ndarray) -> np.ndarray:
        """
        Demodulates received QPSK symbols back to a binary bitstream.

        Args:
            received_signal (np.ndarray): Received complex symbols.
        Returns:
            np.ndarray: Flattened binary bitstream.
        """
        # Use minimum Euclidean distance detection.
        distances = np.abs(received_signal[:, None] - self.constellation)
        symbol_indices = np.argmin(distances, axis=1)
        return self.symbols_to_bits(symbol_indices)

    def bits_to_symbols(self, bits: np.ndarray) -> np.ndarray:
        """
        Converts a binary bitstream to symbol indices using Gray encoding.

        The 2-bit chunks are interpreted in big-endian order, converted to binary
        values, and then mapped to their Gray-coded equivalent.

        For example, the bit chunk:
            [0, 0] -> binary 0 -> Gray 0 -> symbol index 0
            [0, 1] -> binary 1 -> Gray 1 -> symbol index 1
            [1, 0] -> binary 2 -> Gray 3 -> symbol index 3
            [1, 1] -> binary 3 -> Gray 2 -> symbol index 2

        Args:
            bits (np.ndarray): 1D array of bits.
        Returns:
            np.ndarray: Array of symbol indices.
        """
        assert bits.size % self.bits_per_symbol == 0, (
            "Bitstream length must be a multiple of 2 for QPSK."
        )
        bit_chunks = bits.reshape(-1, self.bits_per_symbol)
        symbol_indices = []
        for chunk in bit_chunks:
            # Interpret bits as a big-endian binary number.
            bin_val = (int(chunk[0]) << 1) | int(chunk[1])
            # Convert binary value to Gray code.
            gray_val = self.binary_to_gray(bin_val)
            symbol_indices.append(gray_val)
        return np.array(symbol_indices)

    def symbols_to_bits(self, symbols: np.ndarray) -> np.ndarray:
        """
        Converts symbol indices back to a binary bitstream by inverting the Gray encoding.

        Each symbol index is first converted from its Gray-coded value back to the binary number.
        Then, the binary number is converted to a 2-bit big-endian representation.

        Args:
            symbols (np.ndarray): Array of symbol indices.
        Returns:
            np.ndarray: Flattened binary bitstream.
        """
        bits_list = []
        for sym in symbols:
            # Invert Gray code to recover the binary value.
            bin_val = self.gray_to_binary(sym)
            # Convert the binary value to a 2-bit array (big-endian).
            bits_list.append((bin_val >> 1) & 1)
            bits_list.append(bin_val & 1)
        return np.array(bits_list, dtype=int)

    @staticmethod
    def binary_to_gray(n: int) -> int:
        """
        Converts a binary integer to its Gray code representation.

        Args:
            n (int): Binary integer.
        Returns:
            int: Gray-coded integer.
        """
        return n ^ (n >> 1)

    @staticmethod
    def gray_to_binary(g: int) -> int:
        """
        Converts a Gray-coded integer back to its binary representation.

        Args:
            g (int): Gray-coded integer.
        Returns:
            int: Binary integer.
        """
        b = g
        shift = 1
        while (g >> shift) > 0:
            b ^= (g >> shift)
            shift += 1
        return b


class MPAMModulation(AbstractModulation):
    def __init__(self, M: int):
        """
        Initializes M-PAM modulation for d-ary input.

        Args:
            M (int): Modulation order (e.g., 4 for 4-PAM).
        """
        assert M > 1 and M % 2 == 0, "M must be an even integer >= 2."
        self.M = M
        self.symbols = np.linspace(-M + 1, M - 1, M)

    def modulate(self, input_data: np.ndarray) -> np.ndarray:
        """
        Maps d-ary input to M-PAM symbols.
        """
        if np.any(input_data >= self.M) or np.any(input_data < 0):
            raise ValueError("Input data must be in the range [0, M-1].")
        return self.symbols[input_data.astype(int)]

    def demodulate(self, received_signal: np.ndarray) -> np.ndarray:
        """
        Demodulates M-PAM signal back to d-ary input.
        """
        distances = np.abs(received_signal[:, None] - self.symbols)
        return np.argmin(distances, axis=1).astype(int)


class MQAMModulation(AbstractModulation):
    def __init__(self, M: int, power: float = 1.0):
        """
        Initializes M-QAM modulation with a complex-valued constellation.

        Args:
            M (int): Modulation order (must be a perfect square, e.g., 4, 16, 64).
            power (float): Average power of the constellation.
        """
        # Ensure that M is a power of 2.
        assert (M & (M - 1)) == 0, "M must be a power of 2 (e.g., 4, 16, 64)."
        self.M = M
        self.sqrt_M = int(M ** 0.5)
        # Ensure that M is a perfect square.
        assert self.sqrt_M * self.sqrt_M == M, "M must be a perfect square for QAM."
        self.bits_per_symbol = int(np.log2(M))
        self.power = power
        self.constellation = self._generate_constellation()

    def _generate_constellation(self) -> np.ndarray:
        """
        Generates the complex-valued QAM constellation.
        """
        points = np.arange(self.sqrt_M) * 2 - (self.sqrt_M - 1)
        I, Q = np.meshgrid(points, points)
        constellation = I.flatten() + 1j * Q.flatten()
        E_avg = np.mean(np.abs(constellation) ** 2)
        normalization_factor = np.sqrt(self.power / E_avg)
        return constellation * normalization_factor

    def modulate(self, input_bits: np.ndarray) -> np.ndarray:
        """
        Maps input bits to QAM constellation points.

        Args:
            input_bits (np.ndarray): Binary bitstream (assumed to be pre-reshaped).
        Returns:
            np.ndarray: Complex-valued constellation points.
        """
        symbols = self.bits_to_symbols(input_bits)
        return self.constellation[symbols]

    def demodulate(self, received_signal: np.ndarray) -> np.ndarray:
        """
        Demodulates received QAM symbols back to a binary bitstream.

        Args:
            received_signal (np.ndarray): Received complex symbols.
        Returns:
            np.ndarray: Binary bitstream.
        """
        distances = np.abs(received_signal[:, None] - self.constellation)
        symbols = np.argmin(distances, axis=1)
        return self.symbols_to_bits(symbols)

    def symbols_to_bits(self, symbols: np.ndarray) -> np.ndarray:
        """
        Converts symbols to a binary bitstream in little-endian order.

        Args:
            symbols (np.ndarray): Array of symbol indices.
        Returns:
            np.ndarray: Flattened binary bitstream.
        """
        bits = (symbols[:, None] >> np.arange(self.bits_per_symbol)) & 1
        return bits.flatten()

    def bits_to_symbols(self, bits: np.ndarray) -> np.ndarray:
        """
        Converts a binary bitstream to symbols in little-endian order.

        Args:
            bits (np.ndarray): Binary bitstream.
        Returns:
            np.ndarray: Array of symbol indices.
        """
        # Assumes that the bits array is already reshaped appropriately outside this function.
        bit_chunks = bits.reshape(-1, self.bits_per_symbol)
        return np.sum(bit_chunks * (2 ** np.arange(self.bits_per_symbol)), axis=1)  

class MQUAM(AbstractModulation):
    def __init__(self, M: int, power: float = 1.0):
        """
        Initializes M-QAM modulation with Gray encoding on each dimension.
        
        Args:
            M (int): Modulation order (must be a perfect square, e.g., 4, 16, 64).
            power (float): Average power of the constellation.
        """
        # Check that M is a power of 2.
        assert (M & (M - 1)) == 0, "M must be a power of 2 (e.g., 4, 16, 64)."
        self.M = M
        self.sqrt_M = int(np.sqrt(M))
        # Ensure that M is a perfect square.
        assert self.sqrt_M * self.sqrt_M == M, "M must be a perfect square for QAM."
        self.bits_per_symbol = int(np.log2(M))
        self.power = power
        self.constellation = self._generate_constellation()

    def _generate_constellation(self) -> np.ndarray:
        """
        Generates the complex-valued QAM constellation in natural order.
        
        The constellation is generated as a grid of points in the I-Q plane.
        """
        points = np.arange(self.sqrt_M) * 2 - (self.sqrt_M - 1)
        I, Q = np.meshgrid(points, points)
        constellation = I.flatten() + 1j * Q.flatten()
        E_avg = np.mean(np.abs(constellation) ** 2)
        normalization_factor = np.sqrt(self.power / E_avg)
        return constellation * normalization_factor

    def modulate(self, input_bits: np.ndarray) -> np.ndarray:
        """
        Maps input bits to QAM constellation points using Gray encoding.
        
        Args:
            input_bits (np.ndarray): Binary bitstream. The length must be a multiple of bits_per_symbol.
        Returns:
            np.ndarray: Complex-valued constellation points.
        """
        symbols = self.bits_to_symbols(input_bits)
        return self.constellation[symbols]

    def demodulate(self, received_signal: np.ndarray) -> np.ndarray:
        """
        Demodulates received QAM symbols back to a binary bitstream.
        
        Args:
            received_signal (np.ndarray): Received complex symbols.
        Returns:
            np.ndarray: Binary bitstream.
        """
        distances = np.abs(received_signal[:, None] - self.constellation)
        symbols = np.argmin(distances, axis=1)
        return self.symbols_to_bits(symbols)

    def bits_to_symbols(self, bits: np.ndarray) -> np.ndarray:
        """
        Converts a binary bitstream to symbol indices using Gray encoding.
        
        The input bits are split equally into two parts. Each part is interpreted as a
        big-endian binary number, then converted to its Gray code equivalent. The Gray-coded
        value for the I and Q components is then used to index the constellation grid:
        
            symbol_index = q_gray * sqrt_M + i_gray
        
        Args:
            bits (np.ndarray): 1D binary bitstream.
        Returns:
            np.ndarray: Array of symbol indices.
        """
        bit_chunks = bits.reshape(-1, self.bits_per_symbol)
        dim = self.bits_per_symbol // 2  # number of bits per PAM dimension
        symbol_indices = []
        for chunk in bit_chunks:
            # Split bits: first half for I, second half for Q.
            bits_i = chunk[:dim]
            bits_q = chunk[dim:]
            # Convert bits (big-endian) to integer.
            bin_i = 0
            for b in bits_i:
                bin_i = (bin_i << 1) | int(b)
            bin_q = 0
            for b in bits_q:
                bin_q = (bin_q << 1) | int(b)
            # Apply Gray encoding.
            i_gray = self.binary_to_gray(bin_i)
            q_gray = self.binary_to_gray(bin_q)
            # Combine the two Gray-coded coordinates.
            # Note: The constellation (flattened in row-major order) has row index corresponding to Q.
            symbol_index = q_gray * self.sqrt_M + i_gray
            symbol_indices.append(symbol_index)
        return np.array(symbol_indices)

    def symbols_to_bits(self, symbols: np.ndarray) -> np.ndarray:
        """
        Converts symbol indices back to a binary bitstream by inverting the Gray encoding.
        
        The symbol index is first separated into its Q and I components (recall that
        symbol_index = q_gray * sqrt_M + i_gray). Each component is converted from its Gray code
        back to the binary number. Then each number is converted to a fixed-length bit array (big-endian)
        and concatenated.
        
        Args:
            symbols (np.ndarray): Array of symbol indices.
        Returns:
            np.ndarray: Flattened binary bitstream.
        """
        dim = self.bits_per_symbol // 2  # number of bits per PAM dimension
        bits_list = []
        for sym in symbols:
            # Recover Gray-coded coordinates.
            q_gray = sym // self.sqrt_M
            i_gray = sym % self.sqrt_M
            # Convert Gray code back to binary.
            bin_i = self.gray_to_binary(i_gray)
            bin_q = self.gray_to_binary(q_gray)
            # Convert the binary numbers to bit arrays (big-endian).
            bits_i = [(bin_i >> (dim - 1 - i)) & 1 for i in range(dim)]
            bits_q = [(bin_q >> (dim - 1 - i)) & 1 for i in range(dim)]
            bits_list.extend(bits_i + bits_q)
        return np.array(bits_list, dtype=int)

    @staticmethod
    def binary_to_gray(n: int) -> int:
        """
        Converts a binary integer to its Gray code representation.
        
        Args:
            n (int): Binary integer.
        Returns:
            int: Gray-coded integer.
        """
        return n ^ (n >> 1)

    @staticmethod
    def gray_to_binary(g: int) -> int:
        """
        Converts a Gray-coded integer back to its binary representation.
        
        Args:
            g (int): Gray-coded integer.
        Returns:
            int: Binary integer.
        """
        b = g
        shift = 1
        while (g >> shift) > 0:
            b ^= (g >> shift)
            shift += 1
        return b


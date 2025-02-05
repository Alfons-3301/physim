#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from functools import lru_cache

class MQUAMAntiGray:
    """
    M-QAM modulation with an explicitly defined anti Gray mapping on each PAM dimension.
    
    For a square QAM constellation of order M, the bits-per-symbol are split equally
    between the I and Q dimensions. Instead of standard Gray coding, the mapping for each
    PAM dimension is computed via an anti Gray code algorithm, which is designed to maximize
    the Hamming distance between neighboring labels.
    """

    def __init__(self, M: int, power: float = 1.0):
        # Check that M is a power of 2.
        assert (M & (M - 1)) == 0, "M must be a power of 2 (e.g., 4, 16, 64)."
        self.M = M
        self.sqrt_M = int(np.sqrt(M))
        # Ensure that M is a perfect square.
        assert self.sqrt_M * self.sqrt_M == M, "M must be a perfect square for QAM."
        self.bits_per_symbol = int(np.log2(M))
        self.power = power
        # Each PAM dimension uses half of the bits.
        self.dim = self.bits_per_symbol // 2

        # Compute the anti Gray mapping for one PAM dimension using our explicit function.
        self.anti_mapping = self._compute_anti_gray_code(self.dim)
        # Compute the inverse mapping.
        self.inv_anti_mapping = {v: i for i, v in enumerate(self.anti_mapping)}

        self.constellation = self._generate_constellation()

    def _generate_constellation(self) -> np.ndarray:
        """
        Generates a rectangular QAM constellation (in natural order) and normalizes its power.
        """
        points = np.arange(self.sqrt_M) * 2 - (self.sqrt_M - 1)
        I, Q = np.meshgrid(points, points)
        constellation = I.flatten() + 1j * Q.flatten()
        E_avg = np.mean(np.abs(constellation) ** 2)
        normalization_factor = np.sqrt(self.power / E_avg)
        return constellation * normalization_factor

    def modulate(self, input_bits: np.ndarray) -> np.ndarray:
        """
        Maps input bits to constellation points using the anti Gray mapping.
        """
        symbols = self.bits_to_symbols(input_bits)
        return self.constellation[symbols]

    def demodulate(self, received_signal: np.ndarray) -> np.ndarray:
        """
        Demodulates received QAM symbols back to a binary bitstream.
        """
        distances = np.abs(received_signal[:, None] - self.constellation)
        symbols = np.argmin(distances, axis=1)
        return self.symbols_to_bits(symbols)

    def bits_to_symbols(self, bits: np.ndarray) -> np.ndarray:
        """
        Converts a binary bitstream into symbol indices using the anti Gray mapping.
        The bitstream is partitioned into blocks of length 'bits_per_symbol', with the first
        half representing the I component and the second half the Q component (big‑endian).
        Each component is converted to its anti Gray–coded value.
        """
        bit_chunks = bits.reshape(-1, self.bits_per_symbol)
        symbol_indices = []
        for chunk in bit_chunks:
            # Split bits: first half for I, second half for Q.
            bits_i = chunk[:self.dim]
            bits_q = chunk[self.dim:]
            # Convert bit arrays (big‑endian) to integer.
            bin_i = 0
            for b in bits_i:
                bin_i = (bin_i << 1) | int(b)
            bin_q = 0
            for b in bits_q:
                bin_q = (bin_q << 1) | int(b)
            # Map to anti Gray code.
            i_anti = self._anti_gray_code(bin_i, self.dim)
            q_anti = self._anti_gray_code(bin_q, self.dim)
            # Combine the two anti Gray–coded values into a symbol index.
            symbol_index = q_anti * self.sqrt_M + i_anti
            symbol_indices.append(symbol_index)
        return np.array(symbol_indices)

    def symbols_to_bits(self, symbols: np.ndarray) -> np.ndarray:
        """
        Converts symbol indices back to a binary bitstream by inverting the anti Gray mapping.
        The symbol index is separated into Q and I parts, each of which is mapped back to its
        binary value and then converted to a fixed-length bit array (big‑endian).
        """
        bits_list = []
        for sym in symbols:
            q_anti = sym // self.sqrt_M
            i_anti = sym % self.sqrt_M
            # Invert the anti Gray mapping.
            bin_i = self.inv_anti_mapping[i_anti]
            bin_q = self.inv_anti_mapping[q_anti]
            # Convert the integers to bit arrays (big‑endian) of length self.dim.
            bits_i = [(bin_i >> (self.dim - 1 - i)) & 1 for i in range(self.dim)]
            bits_q = [(bin_q >> (self.dim - 1 - i)) & 1 for i in range(self.dim)]
            bits_list.extend(bits_i + bits_q)
        return np.array(bits_list, dtype=int)

    @staticmethod
    @lru_cache(maxsize=None)
    def _compute_gray_code(bits: int) -> list:
        """
        Recursively computes the Gray code list for the given number of bits.
        """
        if bits == 1:
            return [0, 1]
        else:
            smaller = MQUAMAntiGray._compute_gray_code(bits - 1)
            return smaller + [v | (2 ** (bits - 1)) for v in smaller][::-1]

    @staticmethod
    @lru_cache(maxsize=None)
    def _compute_anti_gray_code(bits: int) -> list:
        """
        Computes the anti Gray code mapping for the given number of bits.
        
        The algorithm is as follows:
          1. Compute the Gray code for (bits - 1) bits.
          2. Left-shift all values by 1.
          3. For each value, append both the value and its complement (with respect to the maximum value).
        
        Returns a list of length 2**bits.
        """
        if bits == 1:
            return [0, 1]
        max_int = (2 ** bits) - 1
        res = MQUAMAntiGray._compute_gray_code(bits - 1)
        res = [value << 1 for value in res]  # left shift
        # Duplicate each value and add its complement.
        res = sum([[value, (~value) & max_int] for value in res], [])
        return res

    @staticmethod
    def _anti_gray_code(value: int, bits: int) -> int:
        """
        Returns the anti Gray code for a given binary value and number of bits.
        """
        mapping = MQUAMAntiGray._compute_anti_gray_code(bits)
        return mapping[value]

    @staticmethod
    def _gray_code(value: int, bits: int) -> int:
        """
        Returns the Gray code for a given binary value and number of bits.
        """
        mapping = MQUAMAntiGray._compute_gray_code(bits)
        return mapping[value]


def visualize_decoding_regions(M=16):
    """
    Visualizes the decision (decoding) regions for the anti Gray 16-QAM constellation.
    The Voronoi diagram of the constellation points approximates the decoding regions,
    and each region is labeled with its corresponding bit sequence.
    """
    # Instantiate the anti Gray modulator.
    mod = MQUAMAntiGray(M=M, power=1.0)
    constellation = mod.constellation
    points = np.column_stack((constellation.real, constellation.imag))

    # Compute the Voronoi diagram.
    vor = Voronoi(points)

    # Create the plot.
    fig, ax = plt.subplots(figsize=(8, 8))
    voronoi_plot_2d(vor, ax=ax, show_vertices=False,
                    line_colors='orange', line_width=2, line_alpha=0.6, point_size=2)
    ax.plot(points[:, 0], points[:, 1], 'bo', markersize=8)

    # Label each constellation point with its corresponding bit sequence.
    for i, (x, y) in enumerate(points):
        bits = mod.symbols_to_bits(np.array([i]))
        bit_str = ''.join(str(b) for b in bits)
        ax.text(x, y, f'\n{bit_str}', ha='center', va='center',
                fontsize=10, color='blue', fontweight='bold')
    ax.set_title("Anti Gray 16-QAM Decoding Regions with Bit Labels")
    ax.set_xlabel("In-phase")
    ax.set_ylabel("Quadrature")
    ax.grid(True)
    plt.show()


if __name__ == '__main__':
    visualize_decoding_regions(M=16)

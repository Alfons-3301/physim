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

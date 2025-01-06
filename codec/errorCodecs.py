from codec.baseClasses import *

import numpy as np
import galois
from channels.baseClasses import AbstractCodec

class HammingCode(AbstractCodec):
    def __init__(self, n: int, k: int):
        """
        Initializes a Hamming code.

        Args:
            n (int): Codeword length (e.g., 7 for (7,4) Hamming Code).
            k (int): Message length (e.g., 4 for (7,4) Hamming Code).
        """
        self.n = n  # Codeword length
        self.k = k  # Message length
        self.G = self._generate_generator_matrix()
        self.H = self._generate_parity_check_matrix()
        self.field = galois.GF(2)

    def _generate_generator_matrix(self):
        """
        Generates the generator matrix G for the Hamming code.
        """
        identity = np.eye(self.k, dtype=int)
        parity = self._generate_parity_check_matrix()[:, :self.k].T
        return np.hstack((identity, parity))

    def _generate_parity_check_matrix(self):
        """
        Generates the parity-check matrix H for the Hamming code.
        """
        H = []
        for i in range(1, 2**self.k):
            binary_representation = [int(bit) for bit in np.binary_repr(i, width=self.n)]
            H.append(binary_representation)
        return np.array(H).T[:, :self.n]

    def encode(self, input_bits: np.ndarray) -> np.ndarray:
        """
        Encodes input bits using the Hamming code.

        Args:
            input_bits (np.ndarray): Input bits to encode (shape: [m, k]).

        Returns:
            np.ndarray: Encoded codewords (shape: [m, n]).
        """
        input_bits = self.field(input_bits)
        return (input_bits @ self.G) % 2

    def decode(self, received_signal: np.ndarray) -> np.ndarray:
        """
        Decodes received signal using the Hamming code.

        Args:
            received_signal (np.ndarray): Received noisy codewords (shape: [m, n]).

        Returns:
            np.ndarray: Decoded message bits (shape: [m, k]).
        """
        received_signal = self.field(received_signal)
        syndromes = (received_signal @ self.H.T) % 2

        # Correct errors
        corrected_codewords = received_signal.copy()
        for i, syndrome in enumerate(syndromes):
            if np.any(syndrome):  # Error detected
                error_index = int("".join(map(str, syndrome)), 2) - 1
                if 0 <= error_index < self.n:
                    corrected_codewords[i, error_index] ^= 1  # Flip the bit

        return corrected_codewords[:, :self.k]

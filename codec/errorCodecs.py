from codec.baseClasses import *

import numpy as np
import galois
from codec.baseClasses import AbstractCodec
from util.parser import ReedSolomonAdapter

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




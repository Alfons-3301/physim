from typing import Tuple
import numpy as np

class BitPadder:
    """
    A utility class to pad/unpad an input bit sequence to match a desired block size.
    """

    @staticmethod
    def pad(bits: np.ndarray, block_size: int) -> Tuple[np.ndarray, int]:
        """
        Pads `bits` so that its length is a multiple of `block_size`.

        Args:
            bits (np.ndarray): 1D array of bits to pad.
            block_size (int): The block size to pad to.

        Returns:
            padded_bits (np.ndarray): Padded bit array.
            pad_length (int): Number of 0 bits appended.
        """
        bits = np.atleast_1d(bits)
        length = bits.shape[0]

        # Calculate how many bits we need to pad
        remainder = length % block_size
        if remainder == 0:
            return bits, 0  # No padding needed

        pad_length = block_size - remainder
        padded_bits = np.concatenate((bits, np.zeros(pad_length, dtype=bits.dtype)))
        return padded_bits, pad_length

    @staticmethod
    def unpad(bits: np.ndarray, pad_length: int) -> np.ndarray:
        """
        Removes `pad_length` bits from the end of `bits`.

        Args:
            bits (np.ndarray): 1D array of padded bits.
            pad_length (int): Number of 0 bits appended during padding.

        Returns:
            np.ndarray: Unpadded bit array.
        """
        if pad_length <= 0:
            return bits
        return bits[:-pad_length]

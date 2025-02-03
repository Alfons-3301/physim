from abc import ABC, abstractmethod
import warnings
import numpy as np

class AbstractParser(ABC):
    """
    Abstract base class for parsers.
    Defines methods for parsing bit sequences into blocks and reconstructing sequences.
    """
    @abstractmethod
    def parse(self, bit_sequence: np.ndarray) -> np.ndarray:
        """
        Parses a raw bit sequence into structured blocks.

        Args:
            bit_sequence (np.ndarray): Input bit sequence (1D array).

        Returns:
            np.ndarray: Parsed blocks (2D array).
        """
        pass

    @abstractmethod
    def reconstruct(self, blocks: np.ndarray) -> np.ndarray:
        """
        Reconstructs the original bit sequence from structured blocks.

        Args:
            blocks (np.ndarray): Parsed blocks (2D array).

        Returns:
            np.ndarray: Reconstructed bit sequence (1D array).
        """
        pass

import warnings
import numpy as np

class BlockParser(AbstractParser):
    """
    Concrete implementation of the AbstractParser.
    Parses bit sequences into fixed-size blocks and reconstructs them.
    """
    def __init__(self, block_size: int):
        """
        Initializes the BlockParser with a fixed block size.

        Args:
            block_size (int): Number of bits per block.
        """
        self.block_size = block_size
        self._padding = None  # Tracks the amount of padding added during parsing

    def parse(self, bit_sequence: np.ndarray) -> np.ndarray:
        """
        Parses the bit sequence into fixed-size blocks.

        Args:
            bit_sequence (np.ndarray): Input bit sequence (1D array).

        Returns:
            np.ndarray: Parsed blocks (2D array).
        """
        # Calculate padding needed to make the sequence divisible by block size
        padding_needed = (self.block_size - len(bit_sequence) % self.block_size) % self.block_size

        # Check for changes in padding
        if self._padding is not None and self._padding != padding_needed:
            warnings.warn(
                f"Padding amount has changed from {self._padding} to {padding_needed}."
            )

        # Update the padding attribute
        self._padding = padding_needed

        # Add padding to the sequence
        padded_sequence = np.pad(bit_sequence, (0, padding_needed), constant_values=0)

        # Reshape into blocks
        return padded_sequence.reshape(-1, self.block_size)

    def reconstruct(self, blocks: np.ndarray) -> np.ndarray:
        """
        Reconstructs the bit sequence from blocks, removing any padding.

        Args:
            blocks (np.ndarray): Parsed blocks (2D array).

        Returns:
            np.ndarray: Reconstructed bit sequence (1D array).
        """
        # Check if parse has been called
        if self._padding is None:
            warnings.warn("The `parse` method has not been called. Cannot remove padding.")
            return blocks.flatten()

        # Flatten the blocks into a 1D array
        reconstructed_sequence = blocks.flatten()

        # Remove the exact number of padded bits
        if self._padding > 0:
            return reconstructed_sequence[:-self._padding]
        else:
            return reconstructed_sequence

    
class ReedSolomonAdapter(AbstractParser):
    """
    Adapts symbol arrays to work with the ReedSolomonCodec.
    Handles reshaping, padding, and unpadding for encoding and decoding.
    """
    def __init__(self, k: int):
        """
        Initializes the ReedSolomonAdapter with the message length.

        Args:
            k (int): Message length for Reed-Solomon encoding (number of symbols per codeword).
        """
        self.k = k
        self._padding = None  # Tracks the amount of padding added during parsing

    def parse(self, symbols: np.ndarray) -> np.ndarray:
        """
        Adapts the symbol array for Reed-Solomon encoding.

        Args:
            symbols (np.ndarray): Input symbol array (1D array).

        Returns:
            np.ndarray: Adapted symbol array (2D array) with shape [N, k].
        """
        # Calculate padding needed to make the number of symbols divisible by k
        num_symbols = len(symbols)
        padding_needed = (self.k - num_symbols % self.k) % self.k

        # Check for changes in padding
        if self._padding is not None:
            if self._padding != padding_needed:
                warnings.warn(
                    f"Padding amount has changed from {self._padding} to {padding_needed}."
                )

        # Update padding attribute
        self._padding = padding_needed

        # Add padding (zero symbols)
        padded_symbols = np.pad(symbols, (0, padding_needed), constant_values=0)

        # Reshape into [N, k]
        return padded_symbols.reshape(-1, self.k)

    def reconstruct(self, encoded_symbols: np.ndarray) -> np.ndarray:
        """
        Adapts the decoded symbol array back into its original shape.

        Args:
            encoded_symbols (np.ndarray): Encoded symbol array (2D array) with shape [N, k].

        Returns:
            np.ndarray: Flattened symbol array (1D array), with padding removed.
        """
        # Check if parse has been called
        if self._padding is None:
            warnings.warn("The `parse` method has not been called. Cannot remove padding.")
            return encoded_symbols.flatten()

        # Flatten the encoded symbols back into a 1D array
        flattened_symbols = encoded_symbols.flatten()

        # Remove the exact number of padded zeros
        if self._padding > 0:
            return flattened_symbols[:-self._padding]
        else:
            return flattened_symbols



# Example Usage
if __name__ == "__main__":
    # Input bit sequence
    bit_sequence = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1])

    # Create a BlockParser with block size of 4
    parser = BlockParser(block_size=7)

    # Parse the bit sequence into blocks
    blocks = parser.parse(bit_sequence)
    print("Parsed Blocks:\n", blocks)
    print(f"Shape of Parsed Blocks: {blocks.shape}")

    # Reconstruct the bit sequence
    reconstructed_sequence = parser.reconstruct(blocks)
    print("Reconstructed Sequence:\n", reconstructed_sequence)

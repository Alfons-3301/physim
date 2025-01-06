from abc import ABC, abstractmethod
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
        reconstructed_sequence = blocks.flatten()

        # Remove trailing padding (zeros)
        return reconstructed_sequence[:np.where(reconstructed_sequence != 0)[0][-1] + 1]

# Example Usage
if __name__ == "__main__":
    # Input bit sequence
    bit_sequence = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1])

    # Create a BlockParser with block size of 4
    parser = BlockParser(block_size=7)

    # Parse the bit sequence into blocks
    blocks = parser.parse(bit_sequence)
    print("Parsed Blocks:\n", blocks)

    # Reconstruct the bit sequence
    reconstructed_sequence = parser.reconstruct(blocks)
    print("Reconstructed Sequence:\n", reconstructed_sequence)

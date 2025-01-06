from abc import ABC, abstractmethod
import numpy as np

class AbstractCodec(ABC):
    @abstractmethod
    def encode(self, input_bits: np.ndarray) -> np.ndarray:
        """
        Encodes the input bits into a codeword.
        """
        pass

    @abstractmethod
    def decode(self, received_signal: np.ndarray) -> np.ndarray:
        """
        Decodes the received signal back into the original input bits.
        """
        pass

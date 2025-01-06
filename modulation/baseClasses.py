from abc import ABC, abstractmethod
import numpy as np

class AbstractModulation(ABC):
    @abstractmethod
    def modulate(self, input_data: np.ndarray) -> np.ndarray:
        """
        Modulates input data into a signal.

        Args:
            input_data (np.ndarray): Input data to modulate (e.g., binary bits).

        Returns:
            np.ndarray: Modulated signal.
        """
        pass

    @abstractmethod
    def demodulate(self, received_signal: np.ndarray) -> np.ndarray:
        """
        Demodulates received signal back into input data.

        Args:
            received_signal (np.ndarray): Received modulated signal.

        Returns:
            np.ndarray: Demodulated data.
        """
        pass

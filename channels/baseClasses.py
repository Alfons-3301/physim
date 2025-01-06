import numpy as np
from abc import ABC, abstractmethod

class AbstractChannel(ABC):
    @abstractmethod
    def transmit(self, input_signal: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def set_parameters(self, params: dict):
        pass
    
class AbstractDiscreteChannel(AbstractChannel):
    @abstractmethod
    def transmit(self, input_bits: np.ndarray) -> np.ndarray:
        """
        Simulates transmission over a discrete channel (e.g., Binary Symmetric Channel).
        """
        pass
    
class AbstractContinuousChannel(AbstractChannel):
    @abstractmethod
    def transmit(self, input_signal: np.ndarray) -> np.ndarray:
        """
        Simulates transmission over a continuous channel (e.g., AWGN).
        """
        pass

class AbstractFadingChannel(ABC):
    @abstractmethod
    def transmit(self, input_signal: np.ndarray) -> np.ndarray:
        """
        Simulates transmission through a fading channel.
        """
        pass

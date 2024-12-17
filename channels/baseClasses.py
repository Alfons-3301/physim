import torch
from abc import ABC, abstractmethod

class AbstractChannel(ABC):
    @abstractmethod
    def transmit(self, input_signal: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def set_parameters(self, params: dict):
        pass
    
class AbstractDiscreteChannel(AbstractChannel):
    @abstractmethod
    def transmit(self, input_bits: torch.Tensor) -> torch.Tensor:
        """
        Simulates transmission over a discrete channel (e.g., Binary Symmetric Channel).
        """
        pass
    
class AbstractContinuousChannel(AbstractChannel):
    @abstractmethod
    def transmit(self, input_signal: torch.Tensor) -> torch.Tensor:
        """
        Simulates transmission over a continuous channel (e.g., AWGN).
        """
        pass



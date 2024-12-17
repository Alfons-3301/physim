from abc import ABC, abstractmethod
import torch

class AbstractModulation(ABC):
    @abstractmethod
    def modulate(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Modulates input data into a signal.
        """
        pass

    @abstractmethod
    def demodulate(self, received_signal: torch.Tensor) -> torch.Tensor:
        """
        Demodulates received signal back into input data.
        """
        pass

from modulation.baseClasses import *

class BPSKModulation(AbstractModulation):
    def modulate(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Modulates binary input {0, 1} to BPSK symbols {-1, +1}.
        """
        return 2 * input_data - 1.0  # Map: 0 -> -1, 1 -> +1

    def demodulate(self, received_signal: torch.Tensor) -> torch.Tensor:
        """
        Demodulates BPSK symbols back to binary {0, 1}.
        """
        return (received_signal > 0).float()  # Threshold at 0
    
class MPAMModulation(AbstractModulation):
    def __init__(self, M: int):
        """
        Initializes M-PAM modulation for d-ary input.

        Args:
            M (int): Modulation order (e.g., 4 for 4-PAM).
        """
        assert M > 1 and M % 2 == 0, "M must be an even integer >= 2."
        self.M = M
        self.symbols = torch.linspace(-M + 1, M - 1, steps=M)

    def modulate(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Maps d-ary input to M-PAM symbols.
        """
        if torch.any(input_data >= self.M) or torch.any(input_data < 0):
            raise ValueError("Input data must be in the range [0, M-1].")
        return self.symbols[input_data.long()]

    def demodulate(self, received_signal: torch.Tensor) -> torch.Tensor:
        """
        Demodulates M-PAM signal back to d-ary input.
        """
        distances = torch.abs(received_signal.unsqueeze(1) - self.symbols)
        return torch.argmin(distances, dim=1).float()

# MQAM Modulation Implementation
class MQAMModulation(AbstractModulation):
    def __init__(self, M: int, power: float = 1.0):
        assert (M & (M - 1)) == 0, "M must be a power of 2 (e.g., 4, 16, 64)."
        self.M = M
        self.sqrt_M = int(M ** 0.5)
        self.bits_per_symbol = int(torch.log2(torch.tensor(M,dtype=torch.float32)))
        self.power = power
        self.constellation = self._generate_constellation()

    def _generate_constellation(self) -> torch.Tensor:
        points = torch.arange(self.sqrt_M) * 2 - (self.sqrt_M - 1)
        I, Q = torch.meshgrid(points, points)  # Compatibility fix
        constellation = torch.stack((I.flatten(), Q.flatten()), dim=1).float()
        E_avg = torch.mean(torch.sum(constellation ** 2, dim=1))
        normalization_factor = torch.sqrt(self.power / E_avg)
        return constellation * normalization_factor

    def modulate(self, input_symbols: torch.Tensor) -> torch.Tensor:
        return self.constellation[input_symbols.long()]

    def demodulate(self, received_signal: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(received_signal, self.constellation)
        return torch.argmin(distances, dim=1)

    def symbols_to_bits(self, symbols: torch.Tensor) -> torch.Tensor:
        """Converts symbols to binary bitstream."""
        return ((symbols.unsqueeze(1) >> torch.arange(self.bits_per_symbol)) & 1).flatten().float()

    def bits_to_symbols(self, bits: torch.Tensor) -> torch.Tensor:
        """Converts a bitstream to symbols."""
        bit_chunks = bits.reshape(-1, self.bits_per_symbol)
        return torch.sum(bit_chunks * (2 ** torch.arange(self.bits_per_symbol)), dim=1)

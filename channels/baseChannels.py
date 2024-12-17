from channels.baseClasses import *

class BinarySymmetricChannel(AbstractDiscreteChannel):
    def __init__(self, error_probability: float):
        self.error_probability = error_probability

    def transmit(self, input_bits: torch.Tensor) -> torch.Tensor:
        noise = torch.bernoulli(torch.full_like(input_bits, self.error_probability))
        return (input_bits + noise) % 2  # XOR operation for BSC

    def set_parameters(self, params: dict):
        self.error_probability = params.get("error_probability", self.error_probability)

class AWGNChannel(AbstractContinuousChannel):
    def __init__(self, snr_db: float):
        self.snr_db = snr_db
        self.noise_variance = 10 ** (-snr_db / 10) #assumes P_signal=1.

    def transmit(self, input_signal: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(input_signal) * torch.sqrt(torch.tensor(self.noise_variance))
        return input_signal + noise

    def set_parameters(self, params: dict):
        self.snr_db = params.get("snr_db", self.snr_db)
        self.noise_variance = 10 ** (-self.snr_db / 10)
        
# BAWGN Channel Implementation
class BAWGNChannel(AbstractContinuousChannel):
    def __init__(self, snr_db: float):
        """
        Binary Additive White Gaussian Noise (BAWGN) Channel.

        Args:
            snr_db (float): Signal-to-noise ratio in decibels.
        """
        self.snr_db = snr_db
        self.noise_variance = self._calculate_noise_variance(snr_db)

    def _calculate_noise_variance(self, snr_db: float) -> float:
        """
        Converts SNR in dB to noise variance.

        Args:
            snr_db (float): SNR in dB.
        
        Returns:
            float: Noise variance (sigma^2).
        """
        snr_linear = 10 ** (snr_db / 10)  # SNR in linear scale
        return 1 / (2 * snr_linear)  # For BPSK signals, sigma^2 = 1 / (2 * SNR)

    def transmit(self, input_bits: torch.Tensor) -> torch.Tensor:
        """
        Simulates BAWGN channel transmission for binary inputs.

        Args:
            input_bits (torch.Tensor): Tensor of binary bits (0 or 1).

        Returns:
            torch.Tensor: Noisy continuous signal.
        """
        # Map binary {0, 1} -> BPSK symbols {-1, +1}
        bpsk_signal = 2 * input_bits - 1.0  # Maps 0 to -1 and 1 to +1

        # Add Gaussian noise
        noise = torch.randn_like(bpsk_signal) * torch.sqrt(torch.tensor(self.noise_variance))
        received_signal = bpsk_signal + noise

        return received_signal

    def set_parameters(self, params: dict):
        """
        Updates the SNR and noise variance.

        Args:
            params (dict): Dictionary with 'snr_db' as key.
        """
        self.snr_db = params.get("snr_db", self.snr_db)
        self.noise_variance = self._calculate_noise_variance(self.snr_db)

    def get_parameters(self) -> dict:
        """
        Returns the current channel parameters.

        Returns:
            dict: Dictionary containing SNR and noise variance.
        """
        return {"snr_db": self.snr_db, "noise_variance": self.noise_variance}



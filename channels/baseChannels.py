from channels.baseClasses import *
import numpy as np

class BinarySymmetricChannel(AbstractDiscreteChannel):
    def __init__(self, error_probability: float):
        self.error_probability = error_probability

    def transmit(self, input_bits: np.ndarray) -> np.ndarray:
        noise = np.random.binomial(1, self.error_probability, size=input_bits.shape)
        return (input_bits + noise) % 2  # XOR operation for BSC

    def set_parameters(self, params: dict):
        self.error_probability = params.get("error_probability", self.error_probability)

class AWGNChannel(AbstractContinuousChannel):
    def __init__(self, snr_db: float):
        self.snr_db = snr_db
        self.noise_variance = 10 ** (-snr_db / 10)  # Assumes P_signal = 1.

    def transmit(self, input_signal: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, np.sqrt(self.noise_variance), size=input_signal.shape)
        return input_signal + noise

    def set_parameters(self, params: dict):
        self.snr_db = params.get("snr_db", self.snr_db)
        self.noise_variance = 10 ** (-self.snr_db / 10)
        
class ComplexAWGNChannel(AbstractContinuousChannel):
    def __init__(self, snr_db: float):
        """
        Initializes a complex AWGN channel.

        Args:
            snr_db (float): Signal-to-noise ratio (SNR) in decibels.
        """
        self.snr_db = snr_db
        self.noise_variance = self._calculate_noise_variance(snr_db)

    def _calculate_noise_variance(self, snr_db: float) -> float:
        """
        Calculates noise variance based on SNR.

        Args:
            snr_db (float): SNR in dB.

        Returns:
            float: Noise variance.
        """
        snr_linear = 10 ** (snr_db / 10)  # SNR in linear scale
        return 1 / (2 * snr_linear)  # For complex signals, variance is halved per dimension

    def transmit(self, input_signal: np.ndarray) -> np.ndarray:
        """
        Adds complex Gaussian noise to the input signal.

        Args:
            input_signal (np.ndarray): Input complex-valued signal.

        Returns:
            np.ndarray: Signal with added noise.
        """
        noise_real = np.random.normal(0, np.sqrt(self.noise_variance), size=input_signal.shape)
        noise_imag = np.random.normal(0, np.sqrt(self.noise_variance), size=input_signal.shape)
        complex_noise = noise_real + 1j * noise_imag
        return input_signal + complex_noise

    def set_parameters(self, params: dict):
        """
        Updates the channel parameters.

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

class PhaseNoiseChannel(AbstractContinuousChannel):
    def __init__(self, sigma: float):
        """
        Initializes the Phase Noise Channel.

        Args:
            sigma (float): Standard deviation of the phase noise in radians.
        """
        self.sigma = sigma

    def set_parameters(self, params: dict):
        """
        Set the parameters of the phase noise channel.
        """
        if 'sigma' in params:
            self.sigma = params['sigma']

    def transmit(self, input_signal: np.ndarray) -> np.ndarray:
        """
        Transmits the input signal through the phase noise channel.

        Args:
            input_signal (np.ndarray): Input QAM symbols (complex-valued).

        Returns:
            np.ndarray: Output symbols with added phase noise.
        """
        # Generate phase noise
        phase_noise = np.random.normal(0, self.sigma, size=(input_signal.shape[0],))
        # Apply phase noise to the input signal
        rotated_signal = input_signal * np.exp(1j * phase_noise)
        return rotated_signal

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

    def transmit(self, input_bits: np.ndarray) -> np.ndarray:
        """
        Simulates BAWGN channel transmission for binary inputs.

        Args:
            input_bits (np.ndarray): Array of binary bits (0 or 1).

        Returns:
            np.ndarray: Noisy continuous signal.
        """
        # Map binary {0, 1} -> BPSK symbols {-1, +1}
        bpsk_signal = 2 * input_bits - 1.0  # Maps 0 to -1 and 1 to +1

        # Add Gaussian noise
        noise = np.random.normal(0, np.sqrt(self.noise_variance), size=bpsk_signal.shape)
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

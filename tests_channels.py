import unittest
import torch

# Import the channel classes
from channels.baseChannels import BinarySymmetricChannel, BAWGNChannel, AWGNChannel

class TestBinarySymmetricChannel(unittest.TestCase):
    def setUp(self):
        self.error_probability = 0.1
        self.channel = BinarySymmetricChannel(self.error_probability)

    def test_transmit_no_noise(self):
        """Test transmission with error_probability = 0 (no noise).\n"""
        self.channel.set_parameters({"error_probability": 0.0})
        input_bits = torch.randint(0, 2, (10, 10)).float()
        output_bits = self.channel.transmit(input_bits)
        self.assertTrue(torch.equal(input_bits, output_bits), "Output should match input with no noise.")

    def test_transmit_with_noise(self):
        """Test transmission with a small error probability.\n"""
        input_bits = torch.zeros(1000).float()  # All 0s input
        output_bits = self.channel.transmit(input_bits)
        flipped_bits = torch.sum(output_bits != input_bits)
        expected_flips = self.error_probability * 1000
        self.assertAlmostEqual(flipped_bits.item(), expected_flips, delta=30, 
                               msg="Flipped bits should approximately match error probability.")

    def test_set_parameters(self):
        """Test updating the error probability.\n"""
        self.channel.set_parameters({"error_probability": 0.2})
        self.assertEqual(self.channel.error_probability, 0.2, "Error probability not updated correctly.")

class TestBAWGNChannel(unittest.TestCase):
    def setUp(self):
        self.snr_db = 10
        self.channel = BAWGNChannel(self.snr_db)

    def test_noise_variance_calculation(self):
        """Test noise variance calculation.\n"""
        expected_variance = 1 / (2 * (10 ** (self.snr_db / 10)))
        self.assertAlmostEqual(self.channel.noise_variance, expected_variance, 
                               places=6, msg="Noise variance not calculated correctly.")

    def test_transmit_signal(self):
        """Test transmission over BAWGN channel.\n"""
        input_bits = torch.tensor([0, 1, 1, 0]).float()
        output_signal = self.channel.transmit(input_bits)
        self.assertEqual(output_signal.shape, input_bits.shape, 
                         "Output shape does not match input shape.")
        self.assertTrue(torch.is_tensor(output_signal), "Output should be a PyTorch tensor.")

    def test_set_parameters(self):
        """Test updating SNR and recalculating noise variance.\n"""
        self.channel.set_parameters({"snr_db": 5})
        new_variance = 1 / (2 * (10 ** (5 / 10)))
        self.assertAlmostEqual(self.channel.noise_variance, new_variance, 
                               places=6, msg="Noise variance not updated correctly.")

class TestAWGNChannel(unittest.TestCase):
    def setUp(self):
        self.snr_db = 10
        self.channel = AWGNChannel(self.snr_db)

    def test_transmit_signal(self):
        """Test transmission over AWGN channel.\n"""
        input_signal = torch.randn(10, 10)
        output_signal = self.channel.transmit(input_signal)
        self.assertEqual(output_signal.shape, input_signal.shape, 
                         "Output shape does not match input shape.")
        self.assertTrue(torch.is_tensor(output_signal), "Output should be a PyTorch tensor.")

    def test_noise_addition(self):
        """Test that noise is added to the input signal.\n"""
        input_signal = torch.zeros(1000)
        output_signal = self.channel.transmit(input_signal)
        noise_std = torch.sqrt(torch.tensor(self.channel.noise_variance))
        measured_std = torch.std(output_signal)
        self.assertAlmostEqual(measured_std.item(), noise_std.item(), delta=0.1, 
                               msg="Noise standard deviation is incorrect.")

    def test_set_parameters(self):
        """Test updating SNR and recalculating noise variance.\n"""
        self.channel.set_parameters({"snr_db": 5})
        new_variance = 10 ** (-5 / 10)
        self.assertAlmostEqual(self.channel.noise_variance, new_variance, 
                               places=6, msg="Noise variance not updated correctly.")

if __name__ == "__main__":
    unittest.main()

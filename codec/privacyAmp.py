import galois
import numpy as np 
from codec.baseClasses import AbstractCodec

class PrivacyAmplification(AbstractCodec):
    def __init__(
        self,
        k: int = 250,  # length of message vector
        q: int = 300,  # length of encoded vector
        gf: galois.GF = galois.GF2,  # Galois Field class
        enable: bool = True,  # if False, just pad zeros to the input
    ):
        super().__init__()
        self.k = k
        self.q = q
        self.GF = gf
        self.enable = enable
        self.b = q - k

        if self.enable:
            self.M = self._generate_matrix()
            self.M_inv = self._invert_matrix(self.M)
        else:
            # When disabled, the matrices are not used.
            self.M = None
            self.M_inv = None

    def _generate_matrix(self) -> np.ndarray:
        """
        Generates a random reversible matrix M with dimensions (q, q).
        """
        M = np.random.randint(0, 2, size=(self.q, self.q))
        M = self.GF(M)
        while np.linalg.matrix_rank(M) < self.q:
            M = np.random.randint(0, 2, size=(self.q, self.q))
            M = self.GF(M)
        return M
    
    def _invert_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Inverts a matrix in the Galois Field.
        """
        M_inv = np.linalg.inv(matrix)
        return self.GF(M_inv)
    
    def _hash(self, V):
        """
        Hashes the input bits to a fixed length.
        """
        assert len(V) == self.q
        prod = V @ self.M
        prod = prod[:self.k]
        return self.GF(prod)
    
    def _dehash(self, message):
        """
        De-hashes the input bits, by concatenating random bits to the input.
        """
        assert len(message) == self.k
        
        B = np.random.randint(0, 2, size=self.b)
        B = self.GF(B)
        concat_bits = np.concatenate((message, B))
        V = concat_bits @ self.M_inv
        return self.GF(V)

    def encode(self, message: np.ndarray) -> np.ndarray:
        """
        Encodes the message using Privacy Amplification.
        If enable is False, simply pads zeros to the input to achieve a blocklength of q.
        """
        assert len(message) == self.k

        if not self.enable:
            # Pad with zeros to reach length q
            pad = np.zeros(self.b, dtype=message.dtype)
            encoded_message = np.concatenate((message, pad))
            return encoded_message
        
        # When enabled, proceed with the normal privacy amplification encoding.
        message = self.GF(message)
        result = self._dehash(message)
        return np.array(result)
 
    def decode(self, received_signal):
        """
        Decodes the received signal using Privacy Amplification.
        If enable is False, simply extracts the first k bits.
        """
        assert len(received_signal) == self.q

        if not self.enable:
            # Extract the first k bits (the original message)
            return received_signal[:self.k]
        
        # When enabled, proceed with the normal privacy amplification decoding.
        received_signal = self.GF(received_signal)
        result = self._hash(received_signal)
        return np.array(result)

    def reset_matrix(self):
        """
        Resets the matrix M to a new random matrix.
        """
        if self.enable:
            self.M = self._generate_matrix()
            self.M_inv = self._invert_matrix(self.M)
        else:
            # When disabled, there is no matrix to reset.
            pass


if __name__ == "__main__":
    # Try both enabled and disabled modes

    # Mode with privacy amplification enabled:
    privacy_amp_enabled = PrivacyAmplification(
        k=10,
        q=20,
        gf=galois.GF2,
        enable=True
    )

    message = np.random.randint(0, 2, size=10)
    print(f"Input Message (enabled):\t{message}")

    encoded_message = privacy_amp_enabled.encode(message)
    print(f"Encoded Message (enabled):\t{encoded_message}")

    decoded_message = privacy_amp_enabled.decode(encoded_message)
    print(f"Decoded Message (enabled):\t{decoded_message}")

    assert np.array_equal(message, decoded_message), "Decoded message does not match input message in enabled mode."

    # Mode with privacy amplification disabled:
    privacy_amp_disabled = PrivacyAmplification(
        k=10,
        q=20,
        gf=galois.GF2,
        enable=False
    )

    print("\n--- Disabled Mode ---")
    print(f"Input Message (disabled):\t{message}")

    encoded_message_disabled = privacy_amp_disabled.encode(message)
    print(f"Encoded Message (disabled):\t{encoded_message_disabled}")

    decoded_message_disabled = privacy_amp_disabled.decode(encoded_message_disabled)
    print(f"Decoded Message (disabled):\t{decoded_message_disabled}")

    assert np.array_equal(message, decoded_message_disabled), "Decoded message does not match input message in disabled mode."

import galois
import numpy as np 
from codec.baseClasses import AbstractCodec

class PrivacyAmplification(AbstractCodec):
    def __init__(
        self,
        k: int = 250, # length of message vector
        q: int = 300, # length of encoded vector
        gf: galois.GF = galois.GF2, # Galois Field class
    ):
        super().__init__()
        self.k = k
        self.q = q
        self.GF = gf
        self.b = q - k

        self.M = self._generate_matrix()
        self.M_inv = self._invert_matrix(self.M)


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
        De-hashes the input bits, by concatinating random bits to input
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
        """
        assert len(message) == self.k

        # cast to Galois Field
        message = self.GF(message)
        
        result = self._dehash(message)

        # cast back to numpy array
        return np.array(result)
 
    def decode(self, received_signal):
        """
        Decodes the received signal using Privacy Amplification.
        """
        assert len(received_signal) == self.q

        # cast to Galois Field
        received_signal = self.GF(received_signal)

        result = self._hash(received_signal)

        # cast back to numpy array
        return np.array(result)

    def reset_matrix(self):
        """
        Resets the matrix M to a new random matrix.
        """
        self.M = self._generate_matrix()
        self.M_inv = self._invert_matrix(self.M)



if __name__ == "__main__":
    

    privacy_amp = PrivacyAmplification(
        k=10,
        q=20,
        gf=galois.GF2,
    )

    message = np.random.randint(0, 2, size=10)
    print(f"Input Message: \t\t{message}")

    encoded_message = privacy_amp.encode(message)
    print(f"Encoded Message: \t{encoded_message}")

    decoded_message = privacy_amp.decode(encoded_message)
    print(f"Decoded Message: \t{decoded_message}")

    assert np.array_equal(message, decoded_message), "Decoded message does not match input message."
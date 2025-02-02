from codec import baseClasses

class ProtocolStack(baseClasses.AbstractCodec):
    """
    Class to support multiple codecs in a stack.
    order of codecs is top to bottom.
    """

    def __init__(self, codecs: list):
        super().__init__()
        self.codecs = codecs

    def encode(self, x):
        for codec in self.codecs:
            x = codec.encode(x)
        return x
    
    def decode(self, y):
        for codec in reversed(self.codecs):
            y = codec.decode(y)
        return y
    

if __name__ == "__main__":
    from codec.privacyAmp import PrivacyAmplification
    from codec.errorCodecs import HammingCode

    # Example usage
    hamming = HammingCode(7, 4)
    privacy_amp = PrivacyAmplification(k=4, q=7)
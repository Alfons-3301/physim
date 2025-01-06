import torch

class BlockParser:
    def __init__(self, block_size: int):
        self.block_size = block_size

    def pad_input(self, input_stream: torch.Tensor, multiple: int = None) -> tuple:
        """
        Pads the input stream to ensure it is divisible by the given multiple.

        Args:
            input_stream (torch.Tensor): Input bitstream (1D tensor).
            multiple (int): The multiple to pad to. Defaults to block_size.

        Returns:
            tuple: (Padded bitstream, Number of padding bits added).
        """
        if multiple is None:
            multiple = self.block_size
        remainder = input_stream.shape[0] % multiple
        if remainder > 0:
            padding_length = multiple - remainder
            padding = torch.zeros(padding_length, dtype=input_stream.dtype)
            return torch.cat((input_stream, padding)), padding_length
        return input_stream, 0

    def split_into_blocks(self, input_stream: torch.Tensor) -> torch.Tensor:
        """
        Splits the input stream into blocks of the desired size.

        Args:
            input_stream (torch.Tensor): Input bitstream (1D tensor).

        Returns:
            torch.Tensor: 2D tensor where each row is a block.
        """
        padded_stream = self.pad_input(input_stream, self.block_size)[0]
        return padded_stream.view(-1, self.block_size)

    def reassemble_stream(self, output_blocks: torch.Tensor, total_length: int) -> torch.Tensor:
        """
        Reassembles the processed blocks back into a continuous bitstream.

        Args:
            output_blocks (torch.Tensor): 2D tensor where each row is a block.
            total_length (int): Original length of the input stream.

        Returns:
            torch.Tensor: Reassembled 1D bitstream truncated to the original length.
        """
        return output_blocks.flatten()[:total_length]

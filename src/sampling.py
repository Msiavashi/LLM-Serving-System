# TODO: The sampling metadata is currently used statically when creating a new Sequence object. 
# This is not ideal because the sampling metadata should be updated during the sampling process. 
# To fix this, we need to update the sampling metadata during the sampling process by adding a 
# new method to the SamplingMetadata class that updates the current token count. We can then 
# call this method in the SequenceBase.

class SamplingMetadata:
    """A class that stores metadata for text generation sampling.

    This class keeps track of the maximum sequence length and the current token count
    during text generation sampling.

    Args:
        num_tokens (int): The maximum number of tokens allowed in the sequence.

    Attributes:
        max_sequence_length (int): The maximum length of tokens allowed in the sequence.
        current_token_count (int): The current number of tokens in the sequence.
    """
    def __init__(self, num_tokens):
        self.max_sequence_length = num_tokens
        self.current_token_count = 0
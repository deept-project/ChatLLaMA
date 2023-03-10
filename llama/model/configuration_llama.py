""" LLaMA configuration"""

import warnings

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "llama-base-cased": "https://huggingface.co/llama-base-cased/resolve/main/config.json",
    "llama-large-cased": "https://huggingface.co/llama-large-cased/resolve/main/config.json",
}


class LLaMAConfig(PretrainedConfig):
    model_type = "llama"
    keys_to_ignore_at_inference = ["mems"]
    attribute_map = {
        "n_token": "vocab_size",  # Backward compatibility
        "hidden_size": "d_model",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        vocab_size: int,
        multiple_of: int,
        norm_eps: float,
        max_batch_size: int,
        max_seq_len: int,
        bos_token_id: int,
        pad_token_id: int,
        eos_token_id: int,
        **kwargs
    ):
        """Constructs LLaMAConfig."""
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

    @property
    def max_position_embeddings(self):
        logger.info(f"The model {self.model_type} is one of the few models that has no sequence length limit.")
        return -1

    @max_position_embeddings.setter
    def max_position_embeddings(self, value):
        # Message copied from Transformer-XL documentation
        raise NotImplementedError(
            f"The model {self.model_type} is one of the few models that has no sequence length limit."
        )
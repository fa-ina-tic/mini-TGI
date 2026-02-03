import logging

import transformers
from transformers import BitsAndBytesConfig
import torch

# Accelerate for multi-GPU/multi-node support
from accelerate import Accelerator, PartialState

logger = logging.getLogger(__name__)

class Generator:
    def __init__(self,
                 model_id = "",
                 continuous_batching: bool = False,
                 dtype = None,
                 trust_remote_code = False,
                 attn_implementation = None,
                 quantization = None,
                 default_seed = None):
        
        # Save configuration
        self.model_id = model_id
        self.continuous_batching = continuous_batching
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.attn_implementation = attn_implementation
        self.quantization = quantization
        self.default_seed = default_seed

        # Initialize Accelerator for multi-GPU/multi-node support
        self.accelerator = Accelerator()
        self.distributed_state = PartialState()

        print(f"Accelerator initialized:")
        print(f"  - Device: {self.accelerator.device}")
        print(f"  - Num processes: {self.accelerator.num_processes}")
        print(f"  - Process index: {self.accelerator.process_index}")
        print(f"  - Is main process: {self.accelerator.is_main_process}")
        print(f"  - Distributed type: {self.accelerator.distributed_type}")

        # Set seed
        if default_seed is not None:
            torch.manual_seed(default_seed)

        # Internal state
        self.model = None
        self.processor = None
        self.running_continuous_batching_manager = None
        self.last_messages = None
        self.last_kv_cache = None

        self.model, self.processor = self._load_model_and_data_processor(self.model_id)
        self.generation_config = self.model.generation_config

    @property
    def device(self) -> torch.device:
        """Returns the device assigned by Accelerator.

        Returns:
            The torch device for this process.
        """
        return self.accelerator.device

    @property
    def is_main_process(self) -> bool:
        """Returns whether this is the main process.

        Returns:
            True if this is the main process, False otherwise.
        """
        return self.accelerator.is_main_process

    def _get_quantization_config(self) -> BitsAndBytesConfig | None:
        """Returns the quantization config based on CLI arguments."""
        if self.quantization == "bnb-4bit":
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.quantization == "bnb-8bit":
            config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            config = None

        if config is not None:
            logger.info(f"Quantization applied: {config}")

        return config

    def _load_model_and_data_processor(self, model_id_and_revision: str):
        """
        Generic method to load a model and a data processor from a model ID and revision, making use of the serve CLI
        arguments.

        Args:
            model_id_and_revision (`str`):
                The model ID and revision to load.
            model_cls (`type[PreTrainedModel]`):
                The model class to load.

        Returns:
            `tuple[PreTrainedModel, Union[ProcessorMixin, PreTrainedTokenizerFast]]`: The loaded model and
            data processor (tokenizer, audio processor, etc.).
        """
        import torch
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer

        logger.info(f"Loading {model_id_and_revision}")

        if "@" in model_id_and_revision:
            model_id, revision = model_id_and_revision.split("@", 1)
        else:
            model_id, revision = model_id_and_revision, "main"

        try:
            data_processor = AutoProcessor.from_pretrained(
                model_id,
                revision=revision,
                trust_remote_code=self.trust_remote_code,
            )
        except OSError:
            try:
                data_processor = AutoTokenizer.from_pretrained(
                    model_id,
                    revision=revision,
                    trust_remote_code=self.trust_remote_code,
                )
            except OSError:
                raise OSError("Failed to load processor with `AutoProcessor` and `AutoTokenizer`.")

        dtype = self.dtype if self.dtype in ["auto", None] else getattr(torch, self.dtype)
        quantization_config = self._get_quantization_config()

        model_kwargs = {
            "revision": revision,
            "attn_implementation": self.attn_implementation,
            "dtype": dtype,
            "device_map": self.device,
            "trust_remote_code": self.trust_remote_code,
            "quantization_config": quantization_config,
        }

        config = AutoConfig.from_pretrained(model_id, **model_kwargs)
        architecture = getattr(transformers, config.architectures[0])
        model = architecture.from_pretrained(model_id, **model_kwargs)

        has_default_max_length = (
            model.generation_config.max_new_tokens is None and model.generation_config.max_length == 20
        )
        has_short_max_new_tokens = (
            model.generation_config.max_new_tokens is not None and model.generation_config.max_new_tokens < 1024
        )
        if has_default_max_length or has_short_max_new_tokens:
            model.generation_config.max_new_tokens = 1024

        logger.info(f"Loaded model {model_id_and_revision}")
        return model, data_processor

if __name__ == "__main__":
    ### Temporal Unit Test Code
    gen = Generator(
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        continuous_batching=True,
    )

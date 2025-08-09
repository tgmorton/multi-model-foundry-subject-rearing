from transformers import AutoConfig, AutoModelForCausalLM
from .config import ExperimentConfig  # Use the new Pydantic model


def create_model(config: ExperimentConfig) -> AutoModelForCausalLM:
    """
    Builds and returns a new GPT-2 model using a validated ExperimentConfig.
    """
    print("--- Building Model from Config ---")

    model_params = config.model
    tokenizer_params = config.tokenizer
    data_params = config.data

    model_config = AutoConfig.from_pretrained("gpt2")

    # Override the base config with parameters from the config object
    model_config.n_layer = model_params.layers
    model_config.n_embd = model_params.embedding_size
    model_config.n_head = model_params.attention_heads
    model_config.n_inner = model_params.intermediate_hidden_size
    model_config.n_positions = data_params.max_sequence_length
    model_config.activation_function = model_params.activation_function
    model_config.resid_pdrop = model_params.dropout
    model_config.attn_pdrop = model_params.attention_dropout
    model_config.vocab_size = tokenizer_params.vocab_size
    model_config.use_cache = False

    print(f"  - Model vocabulary size set to: {model_config.vocab_size}")

    model = AutoModelForCausalLM.from_config(model_config)
    print("  - Successfully created new GPT-2 model with random weights.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Total Parameters: {total_params:,}")

    return model
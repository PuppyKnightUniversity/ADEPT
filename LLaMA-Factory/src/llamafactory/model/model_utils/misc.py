from typing import TYPE_CHECKING

from ...extras import logging
from .visual import COMPOSITE_MODELS
import torch

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer

logger = logging.get_logger(__name__)

def find_all_linear_modules(model: "PreTrainedModel", freeze_vision_tower: bool) -> list[str]:
    r"""Find all available linear modules to apply LoRA, GaLore, or APOLLO."""
    model_type = getattr(model.config, "model_type", None)
    forbidden_modules = {"lm_head"}
    if model_type == "chatglm":
        forbidden_modules.add("output_layer")
    elif model_type == "internlm2":
        forbidden_modules.add("output")

    if model_type in COMPOSITE_MODELS:
        forbidden_modules.add(COMPOSITE_MODELS[model_type].projector_key)

    if freeze_vision_tower and model_type in COMPOSITE_MODELS:
        forbidden_modules.update(COMPOSITE_MODELS[model_type].vision_model_keys)

    module_names = set()
    for name, module in model.named_modules():
        if any(forbidden_module in name for forbidden_module in forbidden_modules):
            continue

        if "Linear" in module.__class__.__name__ and "Embedding" not in module.__class__.__name__:
            module_names.add(name.split(".")[-1])

    logger.info_rank0("Found linear modules: {}".format(",".join(module_names)))
    return list(module_names)

def find_expanded_modules(model: "PreTrainedModel", num_layer_trainable: int) -> list[str]:
    r"""Find the modules in the expanded blocks to apply LoRA."""
    num_layers = getattr(model.config, "num_hidden_layers", None)
    if not num_layers:
        raise ValueError("Model is not supported.")

    # First, check for expanded layers
    expanded_layers = []
    normal_layers = []
    
    # Iterate over model parameters
    for name, param in model.named_parameters():
        # Check if the parameter belongs to down_proj or o_proj
        if "down_proj" in name or "o_proj" in name:
            # Check if the parameter is all zeros (indicating an expanded layer)
            if torch.all(param == 0):
                layer_num = int(''.join(filter(str.isdigit, name.split('.')[2])))  # Extract layer index
                expanded_layers.append(layer_num)
            else:
                layer_num = int(''.join(filter(str.isdigit, name.split('.')[2])))
                normal_layers.append(layer_num)

    # Deduplicate and sort
    expanded_layers = sorted(list(set(expanded_layers)))

    # Determine the strategy based on the number of expanded layers
    if len(expanded_layers) == num_layers:
        # If exactly num_layers expanded layers are found, use them
        trainable_layer_ids = expanded_layers
        logger.info_rank0("Found exactly num_layers expanded layers, using them.")
    elif len(expanded_layers) == 0:
        # If no expanded layers are found, fall back to the original stride-based selection
        if num_layers % num_layer_trainable != 0:
            raise ValueError(
                f"`num_layers` {num_layers} should be divisible by `num_layer_trainable` {num_layer_trainable}."
            )
        stride = num_layers // num_layer_trainable
        trainable_layer_ids = list(range(stride - 1, num_layers + stride - 1, stride))
        logger.info_rank0("No expanded layers found, using original stride-based selection.")
    else:
        # If the number of expanded layers is neither 0 nor num_layers, raise an error
        raise ValueError(
            f"Found {len(expanded_layers)} expanded layers, which is neither 0 nor equal to num_layers ({num_layers}). "
            f"Expanded layers found in layers: {expanded_layers}"
        )

    # Build trainable layer identifiers
    trainable_layers = [f".{idx:d}." for idx in trainable_layer_ids]
    module_names = []
    target_modules = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

    # Find modules that match both the target module types and trainable layers
    for name, _ in model.named_modules():
        if any(target_module in name for target_module in target_modules) and any(
            trainable_layer in name for trainable_layer in trainable_layers
        ):
            module_names.append(name)

    logger.info_rank0("Apply LoRA to layers: {}.".format(",".join(map(str, trainable_layer_ids))))
    return module_names

def find_expanded_modules_ids(model: "PreTrainedModel", num_layer_trainable: int) -> list[str]:
    r"""Identify the indices of expanded layers in the model."""
    num_layers = getattr(model.config, "num_hidden_layers", None)
    if not num_layers:
        raise ValueError("Model is not supported.")

    expanded_layers = []
    
    # Scan parameters to detect zero-initialized (expanded) layers
    for name, param in model.named_parameters():
        if "down_proj" in name or "o_proj" in name:
            if torch.all(param == 0):
                layer_num = int(''.join(filter(str.isdigit, name.split('.')[2])))
                expanded_layers.append(layer_num)

    # Deduplicate and sort layer indices
    expanded_layers = sorted(list(set(expanded_layers)))
    return expanded_layers

def register_autoclass(config: "PretrainedConfig", model: "PreTrainedModel", tokenizer: "PreTrainedTokenizer"):
    """Register model, config, and tokenizer classes for auto-loading if they support it."""
    if "AutoConfig" in getattr(config, "auto_map", {}):
        config.__class__.register_for_auto_class()
    if "AutoModelForCausalLM" in getattr(config, "auto_map", {}):
        model.__class__.register_for_auto_class()
    if "AutoTokenizer" in tokenizer.init_kwargs.get("auto_map", {}):
        tokenizer.__class__.register_for_auto_class()
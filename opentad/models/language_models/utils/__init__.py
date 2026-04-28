from transformers import StoppingCriteria, StoppingCriteriaList
import torch
import torch.nn as nn
import os.path as osp


def enable_gradient_checkpointing(module):
    try:
        module.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        module.gradient_checkpointing_enable()

class StopWordStoppingCriteria(StoppingCriteria):
    """StopWord stopping criteria."""

    def __init__(self, tokenizer, stop_word):
        self.tokenizer = tokenizer
        self.stop_word = stop_word
        self.length = len(self.stop_word)

    def __call__(self, input_ids, *args, **kwargs) -> bool:
        cur_text = self.tokenizer.decode(input_ids[0])
        cur_text = cur_text.replace('\r', '').replace('\n', '')
        return cur_text[-self.length:] == self.stop_word


def get_stop_criteria(
        tokenizer,
        stop_words=[],
):
    stop_criteria = StoppingCriteriaList()
    for word in stop_words:
        stop_criteria.append(StopWordStoppingCriteria(tokenizer, word))
    return stop_criteria

def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    if "output_layer" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("output_layer")
    return list(lora_module_names)

def guess_load_checkpoint(pth_model):
    if osp.isfile(pth_model):
        state_dict = torch.load(pth_model, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
    elif osp.isdir(pth_model):
        try:
            from xtuner.utils.zero_to_any_dtype import (
                get_state_dict_from_zero_checkpoint,
            )
        except ImportError:
            raise ImportError(
                "The provided PTH model appears to be a DeepSpeed checkpoint. "
                "However, DeepSpeed library is not detected in current "
                "environment. This suggests that DeepSpeed may not be "
                "installed or is incorrectly configured. Please verify your "
                "setup."
            )
        state_dict = get_state_dict_from_zero_checkpoint(
            osp.dirname(pth_model), osp.basename(pth_model)
        )
    else:
        raise FileNotFoundError(f"Cannot find {pth_model}")
    return state_dict

def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)

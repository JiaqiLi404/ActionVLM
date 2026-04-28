import os
import torch
import deepspeed
from peft import PeftModel

def ds_custom_save_checkpoint(save_dir, folder, model: deepspeed.DeepSpeedEngine, epoch, optimizer=None, scheduler=None,
                              zero3=False):
    output_dir = os.path.join(save_dir, folder)
    config: dict = model.module.checkpoint_config()
    if zero3:
        state_dict = model._zero3_consolidated_16bit_state_dict()
    else:
        state_dict = model.state_dict()
    if config.get('lora', False):
        lora_state_dict = get_peft_state(
            state_dict.items(), config.get('lora_bias', 'none')
        )
        peft_model = config.get('peft_model')
        peft_model.save_pretrained(output_dir, state_dict=lora_state_dict)

    model_config = config.get('config', None)
    if model_config is not None:
        model_config.save_pretrained(output_dir)

    tokenizer= config.get('tokenizer', None)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)

    specific_module = config.get('specific_module', [])
    specific_module_name = " ".join(specific_module)
    specific_weight_to_save = get_specific_state(state_dict.items(), specific_module)
    torch.save(specific_weight_to_save, os.path.join(output_dir, f'{specific_module_name}.bin')) if len(
        specific_weight_to_save) > 0 else None

    skip_module = config.get('skip_module', [])
    skip_module.extend(specific_module)
    skip_module.append('lora_')
    skip_module = set(skip_module)
    skip_weight_to_save = get_skipped_state(state_dict.items(), skip_module)
    save_states = {
        "epoch": epoch,
        "state_dict": skip_weight_to_save,
    }
    if optimizer is not None:
        save_states.update({"optimizer": optimizer.state_dict()})
    if scheduler is not None:
        save_states.update({"scheduler": scheduler.state_dict()})
    torch.save(save_states, os.path.join(output_dir, f'model.bin')) if len(
        skip_weight_to_save) > 0 else None


def ds_custom_load_checkpoint(save_dir, model: deepspeed.DeepSpeedEngine, optimizer=None, scheduler=None, zero3=False):
    epoch = None
    extra_kwargs = {}
    config: dict = model.module.checkpoint_config()

    model_config = config.get('config', None)
    if model_config is not None:
        model_config.load_pretrained(save_dir)
        extra_kwargs.update({"config": model_config})

    bin_files = [x for x in os.listdir(save_dir) if x.endswith('.bin')]
    for bin_file in bin_files:
        bin_path = os.path.join(save_dir, bin_file)
        state_dict = torch.load(bin_path)
        if optimizer is not None and "optimizer" in state_dict:
            optimizer.load_state_dict(state_dict["optimizer"])
        if scheduler is not None and "scheduler" in state_dict:
            scheduler.load_state_dict(state_dict["scheduler"])
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if "epoch" in state_dict:
            epoch = state_dict["epoch"]
        model.load_state_dict(state_dict, strict=False)

    if config.get('lora', False):
        lora_path = os.path.join(save_dir, 'peft_model')
        model = PeftModel.from_pretrained(model, lora_path)
        extra_kwargs.update({"peft_model": model})

    return model, optimizer, scheduler, epoch, extra_kwargs


def load_checkpoint(save_dir, model, optimizer=None, scheduler=None, custom_save_checkpoint=False, zero3=False):
    epoch = None
    if custom_save_checkpoint and isinstance(model, deepspeed.DeepSpeedEngine):
        model, optimizer, scheduler, epoch, extra_kwargs = ds_custom_load_checkpoint(save_dir, model, optimizer,
                                                                                     scheduler, zero3=zero3)
        if len(extra_kwargs) > 0:
            model.module.load_checkpoint(**extra_kwargs)
    elif custom_save_checkpoint:
        model, optimizer, scheduler, epoch = model.load_checkpoint(save_dir, model)
    elif isinstance(model, deepspeed.DeepSpeedEngine):
        success, client_state = model.load_checkpoint(save_dir)
        if not success:
            raise ValueError(f"Failed to load checkpoint from {save_dir}")
        if "epoch" in client_state:
            epoch = client_state["epoch"]
    else:
        checkpoint = torch.load(save_dir)
        model.load_state_dict(checkpoint["state_dict"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "epoch" in checkpoint:
            epoch = checkpoint["epoch"]

    return model, optimizer, scheduler, epoch

def save_checkpoint(model, model_ema, optimizer, scheduler, epoch, work_dir=None):
    save_dir = os.path.join(work_dir, "checkpoint")

    save_states = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }

    if model_ema != None:
        save_states.update({"state_dict_ema": model_ema.module.state_dict()})

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    checkpoint_path = os.path.join(save_dir, f"epoch_{epoch}.pth")
    torch.save(save_states, checkpoint_path)


def save_best_checkpoint(model, model_ema, epoch, work_dir=None):
    save_dir = os.path.join(work_dir, "checkpoint")

    save_states = {"epoch": epoch, "state_dict": model.state_dict()}

    if model_ema != None:
        save_states.update({"state_dict_ema": model_ema.module.state_dict()})

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    checkpoint_path = os.path.join(save_dir, f"best.pth")
    torch.save(save_states, checkpoint_path)

def load_lora(model, lora_path, non_lora_name='non_lora_trainables.bin'):
    non_lora_trainables_path = os.path.join(lora_path, non_lora_name)
    if os.path.exists(non_lora_trainables_path):
        non_lora_trainables = torch.load(non_lora_trainables_path, map_location='cpu', weights_only=True)
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in
                               non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)
    print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, lora_path)
    return model


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state(named_params, bias="none"):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: v.detach().cpu().clone() for k, v in to_return.items()}
    return to_return


def get_specific_state(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: v.detach().cpu().clone() for k, v in to_return.items()}
    return to_return


def get_skipped_state(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if not any(key_match in k for key_match in keys_to_match)}
    to_return = {k: v.detach().cpu().clone() for k, v in to_return.items()}
    return to_return

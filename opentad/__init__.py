def _patch_transformers_modeling_utils():
    try:
        import torch
        from transformers import modeling_utils, pytorch_utils, utils
    except Exception:
        return

    if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    compat_names = (
        "apply_chunking_to_forward",
        "find_pruneable_heads_and_indices",
        "prune_linear_layer",
    )
    for name in compat_names:
        if not hasattr(modeling_utils, name) and hasattr(pytorch_utils, name):
            setattr(modeling_utils, name, getattr(pytorch_utils, name))

    if not hasattr(utils, "is_flash_attn_greater_or_equal_2_10") and hasattr(utils, "is_flash_attn_greater_or_equal"):
        def is_flash_attn_greater_or_equal_2_10():
            return utils.is_flash_attn_greater_or_equal("2.10")

        utils.is_flash_attn_greater_or_equal_2_10 = is_flash_attn_greater_or_equal_2_10


_patch_transformers_modeling_utils()

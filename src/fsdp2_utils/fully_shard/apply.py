from typing import Any, cast

import torch.nn as nn

from fsdp2_utils.fully_shard.types import FullyShard


def apply_fully_shard(
    module: nn.Module, config: dict[str, Any], recurse: bool = True, pp_enabled: bool = False
):
    """
    Recursively applies fully sharded data parallelism to all child modules before the parent.

    If a module has an attribute '_orig_mod', we still sharding to the compiled module itself,
    not to the original module. The casing on '_orig_mod' is to avoid double-wrapping or
    incorrectly traversing into the original (uncompiled) module.

    Args:
        module (nn.Module): The module to apply sharding to.
        config (dict[str, Any]): Configuration for sharding.
        recurse (bool): Whether to recursively apply to children.
        pp_enabled (bool): Whether Pipeline Parallelism is enabled.
    """
    if pp_enabled:
        # For PP, do not reshard after forward to avoid per-microbatch
        # all-gathers, which can be expensive and non-overlapped
        config["reshard_after_forward"] = False

    # Apply to children first
    if recurse:
        for name, submodule in module.named_children():
            if name == "_orig_mod":
                # Do not recurse into the original module; we want to shard the compiled module itself.
                for name, child in submodule.named_children():
                    apply_fully_shard(child, config, recurse=recurse)
            else:
                apply_fully_shard(submodule, config, recurse=recurse)

    # Then apply to parent if it's parallelizable
    # Protocol check doesnt work for compiled modules
    if hasattr(module, "fully_shard"):
        cast(FullyShard, module).fully_shard(config)  # type: ignore

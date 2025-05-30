from typing import cast

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel.style import ParallelStyle
from torch.distributed.tensor.placement_types import Placement

from fsdp2_utils.tensor_parallel.types import ParallelPlan

from .parallelize import parallelize_module


def construct_plan(module: nn.Module, loss_parallel: bool) -> dict[str, ParallelStyle]:
    # if not isinstance(module, ParallelPlan):
    if not hasattr(module, "parallelize_plan"):
        return {}

    plan_constructor = cast(ParallelPlan, module).parallelize_plan(loss_parallel)
    return {submodule: constructor() for submodule, constructor in plan_constructor.items()}


def apply_tensor_parallel(
    module: nn.Module, tp_mesh: DeviceMesh, recurse: bool = True, loss_parallel: bool = False
) -> nn.Module:
    """Recursively applies tensor parallelism to all child modules first before parent."""
    # Apply to children first
    if recurse:
        for name, child in module.named_children():
            setattr(module, name, apply_tensor_parallel(child, tp_mesh, recurse=recurse))

    # Then apply to parent if it's parallelizable
    # if not isinstance(module, ParallelPlan):
    if not hasattr(module, "parallelize_plan"):
        return module

    parallelize_plan = construct_plan(module, loss_parallel)

    if (input_plan := parallelize_plan.pop("self.input", None)) is not None:
        module = parallelize_module(
            module=module,
            device_mesh=tp_mesh,
            parallelize_plan=input_plan,
        )

    if (output_plan := parallelize_plan.pop("self.output", None)) is not None:
        module = parallelize_module(
            module=module,
            device_mesh=tp_mesh,
            parallelize_plan=output_plan,
        )

    # Extract parameters from parallelize_plan
    # TODO: Allow this to support nested identifiers
    parameter_parallel_plan = {
        name: parallelize_plan.pop(name)
        for name in tuple(parallelize_plan.keys())
        if isinstance(getattr(module, name, None), nn.Parameter)
    }

    for name, placement in parameter_parallel_plan.items():
        parameter = getattr(module, name)

        assert isinstance(placement, Placement), f"Parameter {name} plan must be a placement"
        assert not isinstance(parameter, DTensor), "Don't think this should ever happen"

        module.register_parameter(
            name,
            nn.Parameter(DTensor.from_local(parameter, tp_mesh, [placement], run_check=False)),
        )

    return parallelize_module(
        module=module,
        device_mesh=tp_mesh,
        parallelize_plan=parallelize_plan,
    )

from typing import cast

import torch.nn as nn
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    PrepareModuleOutput,
)
from torch.distributed.tensor.placement_types import Placement, Replicate
from torchao.float8.float8_linear import Float8Linear

from .parallel_style import (
    Float8RowwiseParallel,
    PrepareFloat8Module,
    PrepareModule,
    RowwiseParallel,
    SequenceParallel,
)
from .types import PlanConstructor


def sequence_parallel(
    _: nn.Module,
    *,
    sequence_dim: int = 1,
    input_layouts: Placement | tuple[Placement, ...] | None = None,
    output_layouts: Placement | tuple[Placement, ...] | None = None,
    use_local_output: bool = False,
):
    def _get():
        return SequenceParallel(
            sequence_dim=sequence_dim,
            input_layouts=input_layouts,
            output_layouts=output_layouts,
            use_local_output=use_local_output,
        )

    return cast(PlanConstructor, _get)


def prepare_module_input(
    module: nn.Module,
    *,
    input_layouts: Placement | tuple[Placement | None, ...] | None = None,
    desired_input_layouts: Placement | tuple[Placement | None, ...] | None = None,
    input_kwarg_layouts: dict[str, Placement] | None = None,
    desired_input_kwarg_layouts: dict[str, Placement] | None = None,
    use_local_output: bool = False,
):
    def _get():
        if isinstance(module, Float8Linear) or any(
            isinstance(submodule, Float8Linear) for submodule in module.children()
        ):
            from torchao.float8.float8_tensor_parallel import PrepareFloat8ModuleInput

            return PrepareFloat8ModuleInput(
                input_layouts=input_layouts,
                desired_input_layouts=desired_input_layouts,
                input_kwarg_layouts=input_kwarg_layouts,
                desired_input_kwarg_layouts=desired_input_kwarg_layouts,
                use_local_output=use_local_output,
            )
        return PrepareModuleInput(
            input_layouts=input_layouts,  # type: ignore
            desired_input_layouts=desired_input_layouts,  # type: ignore
            input_kwarg_layouts=input_kwarg_layouts,
            desired_input_kwarg_layouts=desired_input_kwarg_layouts,
            use_local_output=use_local_output,
        )

    return cast(PlanConstructor, _get)


def prepare_module_output(
    _: nn.Module,
    *,
    output_layouts: Placement | tuple[Placement, ...],
    desired_output_layouts: Placement | tuple[Placement, ...],
    use_local_output: bool = True,
):
    def _get():
        return PrepareModuleOutput(
            output_layouts=output_layouts,  # type: ignore
            desired_output_layouts=desired_output_layouts,  # type: ignore
            use_local_output=use_local_output,
        )

    return cast(PlanConstructor, _get)


def colwise_parallel(
    module: nn.Module,
    *,
    input_layouts: Placement | None = None,
    output_layouts: Placement | None = None,
    use_local_output: bool = True,
):
    def _get():
        if isinstance(module, Float8Linear):
            from torchao.float8.float8_tensor_parallel import Float8ColwiseParallel

            return Float8ColwiseParallel(
                input_layouts=input_layouts,
                output_layouts=output_layouts,
                use_local_output=use_local_output,
            )
        return ColwiseParallel(
            input_layouts=input_layouts,
            output_layouts=output_layouts,
            use_local_output=use_local_output,
        )

    return cast(PlanConstructor, _get)


def rowwise_parallel(
    module: nn.Module,
    *,
    input_layouts: Placement | None = None,
    output_layouts: Placement | None = None,
    use_local_output: bool = True,
):
    def _get():
        if isinstance(module, Float8Linear):
            return Float8RowwiseParallel(
                input_layouts=input_layouts,
                output_layouts=output_layouts,
                use_local_output=use_local_output,
            )
        return RowwiseParallel(
            input_layouts=input_layouts,
            output_layouts=output_layouts,
            use_local_output=use_local_output,
        )

    return cast(PlanConstructor, _get)


def parameter_parallel(
    layout: Placement | None = None,
):
    layout = layout or Replicate()

    def _get():
        return Replicate()

    return cast(PlanConstructor, _get)


def prepare_module(
    module: nn.Module,
    *,
    output_layouts: Placement | tuple[Placement, ...],
    desired_output_layouts: Placement | tuple[Placement, ...],
    input_layouts: Placement | tuple[Placement | None, ...] | None = None,
    desired_input_layouts: Placement | tuple[Placement | None, ...] | None = None,
    input_kwarg_layouts: dict[str, Placement] | None = None,
    desired_input_kwarg_layouts: dict[str, Placement] | None = None,
    use_local_output_for_input: bool = False,
    use_local_output_for_output: bool = True,
):
    def _get():
        if isinstance(module, Float8Linear) or any(
            isinstance(submodule, Float8Linear) for submodule in module.children()
        ):
            return PrepareFloat8Module(
                output_layouts=output_layouts,
                desired_output_layouts=desired_output_layouts,
                input_layouts=input_layouts,
                desired_input_layouts=desired_input_layouts,
                input_kwarg_layouts=input_kwarg_layouts,
                desired_input_kwarg_layouts=desired_input_kwarg_layouts,
                use_local_output_for_input=use_local_output_for_input,
                use_local_output_for_output=use_local_output_for_output,
            )
        return PrepareModule(
            output_layouts=output_layouts,  # type: ignore
            desired_output_layouts=desired_output_layouts,  # type: ignore
            input_layouts=input_layouts,  # type: ignore
            desired_input_layouts=desired_input_layouts,  # type: ignore
            input_kwarg_layouts=input_kwarg_layouts,
            desired_input_kwarg_layouts=desired_input_kwarg_layouts,
            use_local_output_for_input=use_local_output_for_input,
            use_local_output_for_output=use_local_output_for_output,
        )

    return cast(PlanConstructor, _get)

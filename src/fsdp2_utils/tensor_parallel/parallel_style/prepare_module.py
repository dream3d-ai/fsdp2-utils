from typing import Dict, Optional, Tuple, Union

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel.style import (
    ParallelStyle,
    PrepareModuleInput,
    PrepareModuleOutput,
)
from torch.distributed.tensor.placement_types import Placement


class PrepareModule(ParallelStyle):
    def __init__(
        self,
        output_layouts: Union[Placement, Tuple[Placement]],
        desired_output_layouts: Union[Placement, Tuple[Placement]],
        input_layouts: Optional[Union[Placement, Tuple[Optional[Placement]]]] = None,
        desired_input_layouts: Optional[Union[Placement, Tuple[Optional[Placement]]]] = None,
        input_kwarg_layouts: Optional[Dict[str, Placement]] = None,
        desired_input_kwarg_layouts: Optional[Dict[str, Placement]] = None,
        use_local_output_for_input: bool = False,
        use_local_output_for_output: bool = True,
    ):
        self.input = PrepareModuleInput(
            input_layouts=input_layouts,
            desired_input_layouts=desired_input_layouts,
            input_kwarg_layouts=input_kwarg_layouts,
            desired_input_kwarg_layouts=desired_input_kwarg_layouts,
            use_local_output=use_local_output_for_input,
        )
        self.output = PrepareModuleOutput(
            output_layouts=output_layouts,
            desired_output_layouts=desired_output_layouts,
            use_local_output=use_local_output_for_output,
        )

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        module = self.input._apply(module, device_mesh)
        module = self.output._apply(module, device_mesh)
        return module


class PrepareFloat8Module(PrepareModule):
    def __init__(self, *args, **kwargs):
        from torchao.float8.float8_tensor_parallel import PrepareFloat8ModuleInput

        super().__init__(*args, **kwargs)
        self.input = PrepareFloat8ModuleInput(
            input_layouts=self.input.input_layouts,
            desired_input_layouts=self.input.desired_input_layouts,
            input_kwarg_layouts=self.input.input_kwarg_layouts,
            desired_input_kwarg_layouts=self.input.desired_input_kwarg_layouts,
            use_local_output=self.input.use_local_output,
        )

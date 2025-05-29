from functools import partial

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, distribute_module
from torch.distributed.tensor.parallel import SequenceParallel as _SequenceParallel
from torch.distributed.tensor.placement_types import Placement, Shard


class SequenceParallel(_SequenceParallel):
    def __init__(
        self,
        *,
        sequence_dim: int = 1,
        input_layouts: Placement | tuple[Placement, ...] | None = None,
        output_layouts: Placement | tuple[Placement, ...] | None = None,
        use_local_output: bool = False,
    ):
        super().__init__(sequence_dim=sequence_dim, use_local_output=use_local_output)
        input_layouts = (
            (input_layouts,) if isinstance(input_layouts, Placement) else input_layouts
        )
        output_layouts = (
            (output_layouts,) if isinstance(output_layouts, Placement) else output_layouts
        )
        self.input_layouts = input_layouts or (Shard(sequence_dim),)
        self.output_layouts = output_layouts or (Shard(sequence_dim),)

    @staticmethod
    def _prepare_input_fn(input_layouts, sequence_sharding, mod, inputs, device_mesh):
        input_tensor = inputs[0]

        if isinstance(input_tensor, DTensor):
            # if the passed in input DTensor is not sharded on the sequence dim, we need to redistribute it
            input_placements = input_layouts or input_tensor.placements
            if input_placements != sequence_sharding:
                input_tensor = input_tensor.redistribute(
                    placements=sequence_sharding, async_op=True
                )
            return input_tensor
        elif isinstance(input_tensor, torch.Tensor):
            if input_layouts is None:
                # assume the input passed in already sharded on the sequence dim and create the DTensor
                return DTensor.from_local(
                    input_tensor,
                    device_mesh,
                    sequence_sharding,
                    run_check=False,
                )
            else:
                input_tensor = DTensor.from_local(
                    input_tensor,
                    device_mesh,
                    input_layouts,
                    run_check=False,
                )
                if input_layouts != sequence_sharding:
                    input_tensor = input_tensor.redistribute(
                        placements=sequence_sharding, async_op=True
                    )
                return input_tensor
        else:
            raise ValueError(
                f"expecting input of {mod} to be a torch.Tensor or DTensor, but got {input_tensor}"
            )

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=True)
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._replicate_module_fn,
            partial(self._prepare_input_fn, self.input_layouts, self.sequence_sharding),  # type: ignore
            partial(self._prepare_output_fn, self.output_layouts, self.use_local_output),
        )

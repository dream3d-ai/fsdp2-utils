import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.parallel import RowwiseParallel as _RowwiseParallel
from torch.distributed.tensor.placement_types import Replicate, Shard
from torchao.float8.float8_tensor_parallel import (
    Float8RowwiseParallel as _Float8RowwiseParallel,
)

"""
RowwiseParallel is commonlly applied to nn.Embedding layers. However, if the number of embeddings is less than the shard size,
pytorch will raise an error. 

This is a custom RowwiseParallel that will replicate the embedding if the number of embeddings is less than the shard size.
"""


class RowwiseParallel(_RowwiseParallel):
    def _partition_embedding_fn(self, name: str, module: nn.Module, device_mesh: DeviceMesh):
        if not isinstance(module, nn.Embedding):
            raise ValueError(f"Module {module} is not an instance of nn.Embedding")

        for name, param in module.named_parameters():
            if param.size(0) < device_mesh.size(0):
                dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Replicate()]))
            else:
                dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            module.register_parameter(name, dist_param)


class Float8RowwiseParallel(_Float8RowwiseParallel):
    def _partition_embedding_fn(self, name: str, module: nn.Module, device_mesh: DeviceMesh):
        if not isinstance(module, nn.Embedding):
            raise ValueError(f"Module {module} is not an instance of nn.Embedding")

        for name, param in module.named_parameters():
            if param.size(0) < device_mesh.size(0):
                dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Replicate()]))
            else:
                dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            module.register_parameter(name, dist_param)

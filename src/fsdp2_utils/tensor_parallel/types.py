from typing import Protocol, runtime_checkable

from torch.distributed.tensor.parallel import ParallelStyle


class PlanConstructor(Protocol):
    def __call__(self, fp8: bool = False) -> ParallelStyle: ...


@runtime_checkable
class ParallelPlan(Protocol):
    def parallel_plan(self, loss_parallel: bool) -> dict[str, PlanConstructor]: ...

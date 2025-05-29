from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class FullyShard(Protocol):
    def fully_shard(self, config: dict[str, Any]) -> None: ...

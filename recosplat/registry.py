"""Name-to-class registry."""

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def decorator(cls: type[T]) -> type[T]:
            if name in self._entries:
                raise KeyError(f"Duplicate {self.kind} registry entry: {name!r}")
            self._entries[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[T]:
        if name not in self._entries:
            raise KeyError(
                f"Unknown {self.kind} {name!r}; registered: {sorted(self._entries)}"
            )
        return self._entries[name]

    def names(self) -> list[str]:
        return sorted(self._entries)

from abc import ABC
from typing import Any, Callable, Optional


class AgentAttribute(ABC):
    def __init__(self, value: Any):
        self._value = value

    @property
    def value(self) -> Any:
        return self._value


class ConstantAgentAttribute(AgentAttribute):
    def change_value(self, delta: Any) -> None:
        raise AttributeError("Cannot change value of ConstantValue")


class MutableAgentAttribute(AgentAttribute):
    def __init__(
            self, value: Any, min_value: Optional[Any] = None, max_value: Optional[Any] = None
    ):
        super().__init__(value)
        self.min = min_value
        self.max = max_value
        self._validate(self._value)

    def _validate(self, value: Any) -> None:
        if self.min is not None and value < self.min:
            raise ValueError(f"Value {value} is below minimum {self.min}")
        if self.max is not None and value > self.max:
            raise ValueError(f"Value {value} is above maximum {self.max}")

    def change_value(self, delta: Any) -> None:
        new_value = self._value + delta
        self._validate(new_value)
        self._value = new_value

    def __add__(self, delta: Any) -> None:
        self.change_value(delta)


class DynamicAgentAttribute(MutableAgentAttribute, ABC):
    def __init__(
            self,
            value: Any,
            min_value: Optional[Any] = None,
            max_value: Optional[Any] = None,
            change_function: Callable[[Any], Any] = lambda x: x,
    ):
        super().__init__(value, min_value, max_value)
        self.change_function = change_function

    def change_value(self) -> None:
        new_value = self.change_function(self._value)
        self._validate(new_value)
        self._value = new_value


class IteratingAgentAttribute(DynamicAgentAttribute):
    def __init__(
            self,
            value: Any,
            min_value: Optional[Any] = None,
            max_value: Optional[Any] = None,
            delta: Any = 1,
    ):
        self.delta = delta
        super().__init__(value, min_value, max_value, self.iterate)

    def iterate(self, current_value: Any) -> Any:
        return current_value + self.delta

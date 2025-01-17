from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, TypeVar, Generic

T = TypeVar('T')


class AgentAttribute(ABC, Generic[T]):
    """
    Abstract base class representing a generic agent attribute.
    Provides a foundation for different types of agent attributes with mechanisms for value management and comparison.
    """

    def __init__(self, value: T):
        self._value = value

    @property
    def value(self) -> T:
        """Gets the current value of the attribute."""
        return self._value

    @value.setter
    def value(self, value: T):
        """Sets the value of the attribute."""
        self._value = value

    @abstractmethod
    def __iadd__(self, delta: Any) -> 'AgentAttribute[T]':
        """Defines in-place addition behavior."""
        pass

    @abstractmethod
    def __isub__(self, delta: Any) -> 'AgentAttribute[T]':
        """Defines in-place subtraction behavior."""
        pass

    @abstractmethod
    def update_on_epoch(self) -> None:
        """Updates the attribute value based on epoch changes."""
        pass

    def __eq__(self, other: Any) -> bool:
        """Checks equality based on the attribute's value."""
        if isinstance(other, AgentAttribute):
            return self._value == other.value
        return NotImplemented

    def __ne__(self, other: Any) -> bool:
        """Checks inequality based on the attribute's value."""
        if isinstance(other, AgentAttribute):
            return self._value != other.value
        return NotImplemented

    def __lt__(self, other: Any) -> bool:
        """Checks if this attribute's value is less than another's."""
        if isinstance(other, AgentAttribute):
            return self._value < other.value
        return NotImplemented

    def __gt__(self, other: Any) -> bool:
        """Checks if this attribute's value is greater than another's."""
        if isinstance(other, AgentAttribute):
            return self._value > other.value
        return NotImplemented

    def __le__(self, other: Any) -> bool:
        """Checks if this attribute's value is less than or equal to another's."""
        if isinstance(other, AgentAttribute):
            return self._value <= other.value
        return NotImplemented

    def __ge__(self, other: Any) -> bool:
        """Checks if this attribute's value is greater than or equal to another's."""
        if isinstance(other, AgentAttribute):
            return self._value >= other.value
        return NotImplemented


class ConstantAgentAttribute(AgentAttribute[T]):
    """
    Represents an agent attribute with a constant value that cannot be modified after initialization.
    """

    def change_value(self, delta: Any) -> None:
        """Attempts to change the attribute's value, but always raises an AttributeError."""
        raise AttributeError("Cannot change value of ConstantAgentAttribute")

    def __iadd__(self, delta: Any) -> 'ConstantAgentAttribute[T]':
        """Prevents in-place addition by raising an AttributeError."""
        raise AttributeError("Cannot modify value of ConstantAgentAttribute")

    def __isub__(self, delta: Any) -> 'ConstantAgentAttribute[T]':
        """Prevents in-place subtraction by raising an AttributeError."""
        raise AttributeError("Cannot modify value of ConstantAgentAttribute")

    def update_on_epoch(self) -> None:
        """Does nothing as the value remains constant across epochs."""
        pass


class MutableAgentAttribute(AgentAttribute[T]):
    """
    Represents an agent attribute whose value can be modified, optionally constrained by minimum and maximum bounds,
    and can change based on epoch updates.
    """

    def __init__(
            self,
            value: T,
            min_value: Optional[T] = None,
            max_value: Optional[T] = None,
            epoch_change_function: Optional[Callable[[T,T,T], T]] = None,
    ):
        super().__init__(value)
        self.min = min_value
        self.max = max_value
        self._validate(self._value)
        self.epoch_change_function = epoch_change_function

    def _validate(self, value: T) -> None:
        """Validates that the value is within the specified bounds."""
        if self.min is not None and value < self.min:
            raise ValueError(f"Value {value} is below minimum {self.min}")
        if self.max is not None and value > self.max:
            raise ValueError(f"Value {value} is above maximum {self.max}")

    def change_value(self, delta: Any) -> None:
        """
        Changes the attribute's value by a specified delta, ensuring it remains within bounds.

        Args:
            delta (Any): The amount to change the value by.
        """
        try:
            new_value = self._value + delta
        except TypeError as e:
            raise TypeError(f"Cannot add delta of type {type(delta)} to value of type {type(self._value)}") from e

        self._validate(new_value)
        self._value = new_value

    def __iadd__(self, delta: Any) -> 'MutableAgentAttribute[T]':
        """Implements in-place addition by modifying the attribute's value."""
        self.change_value(delta)
        return self

    def __isub__(self, delta: Any) -> 'MutableAgentAttribute[T]':
        """Implements in-place subtraction by modifying the attribute's value."""
        self.change_value(-delta)
        return self

    def update_on_epoch(self) -> None:
        """Updates the attribute's value based on the epoch change function, if provided."""
        if self.epoch_change_function:
            new_value = self.epoch_change_function(self._value, self.min, self.max)
            self._validate(new_value)
            self._value = new_value


class IteratingAgentAttribute(MutableAgentAttribute[T]):
    """
    A specialized MutableAgentAttribute that increments its value by a fixed delta on each epoch.
    """

    def __init__(
            self,
            value: T,
            min_value: Optional[T] = None,
            max_value: Optional[T] = None,
            delta: Any = 1,
    ):
        """
        Initializes the iterating attribute with a value, optional bounds, and a fixed delta for iteration.

        Args:
            value (T): The initial value of the attribute.
            min_value (Optional[T]): The minimum allowable value.
            max_value (Optional[T]): The maximum allowable value.
            delta (Any): The fixed amount to add to the value on each epoch.
        """
        self.delta = delta
        super().__init__(value, min_value, max_value, self.iterate)

    def iterate(self, current_value: T, min_value: Optional[T] = None, max_value: Optional[T] = None) -> T:
        """
        Defines the iteration behavior by adding the fixed delta to the current value.

        Args:
            current_value (T): The current value of the attribute.

        Returns:
            T: The new value after adding delta.
            :param max_value:
            :param current_value:
            :param min_value:
        """
        try:
            new_value = current_value + self.delta
        except TypeError as e:
            raise TypeError(
                f"Cannot add delta of type {type(self.delta)} to value of type {type(current_value)}") from e

        if min_value <= new_value <= max_value:
            return new_value
        else:
            return current_value

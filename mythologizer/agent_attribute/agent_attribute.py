from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Optional, TypeVar
from pydantic import BaseModel, Field, ConfigDict, root_validator
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class AgentAttribute(BaseModel, ABC, Generic[T]):
    """
    Abstract base class representing a generic agent attribute.
    Provides a foundation for different types of agent attributes with mechanisms for value management and comparison.
    """
    value: T
    name: str
    description: str
    min: Optional[T] = None
    max: Optional[T] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    def __init__(self, **data: Any):
        super().__init__(**data)
        logger.debug(f"[{self.name}] Initialized {self.__class__.__name__} with data: {data}")

    def __str__(self) -> str:
        attrs = ', '.join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"

    def __setattr__(self, name: str, value: Any):
        if name == 'value':
            self._validate_with_error(value)
            old_value = getattr(self, 'value', None)
            logger.debug(f"[{self.name}] Setting value from {old_value} to {value}")
            super().__setattr__('value', value)
        else:
            error_msg = f"[{self.name}] Only 'value' attribute can be modified."
            logger.error(error_msg)
            raise AttributeError(error_msg)

    def _validate_with_error(self, value: T) -> None:
        """Validates that the value is within the specified bounds."""
        logger.debug(f"[{self.name}] Validating value: {value} with min={self.min} and max={self.max}")
        if self.min is not None and value < self.min:
            error_msg = f"Value {value} is below minimum {self.min}"
            logger.error(f"[{self.name}] {error_msg}")
            raise ValueError(error_msg)
        if self.max is not None and value > self.max:
            error_msg = f"Value {value} is above maximum {self.max}"
            logger.error(f"[{self.name}] {error_msg}")
            raise ValueError(error_msg)

    def _update_with_bounds(self, new_value: T) -> None:
        """Clamps the new_value within min and max bounds and updates the value."""
        original_new_value = new_value
        if self.min is not None and new_value < self.min:
            new_value = self.min
            logger.debug(f"[{self.name}] Value {original_new_value} is below minimum {self.min}. "
                         f"Clamping to {new_value}.")
        if self.max is not None and new_value > self.max:
            new_value = self.max
            logger.debug(f"[{self.name}] Value {original_new_value} is above maximum {self.max}. "
                         f"Clamping to {new_value}.")
        if new_value != self.value:
            old_value = self.value
            super().__setattr__('value', new_value)
            logger.debug(f"[{self.name}] Value changed from {old_value} to {new_value}")
        else:
            logger.debug(f"[{self.name}] New value {new_value} is within bounds; no change made.")

    def _change_value(self, delta: Any) -> None:
        """Changes the value by a given delta, ensuring it stays within bounds."""
        logger.debug(f"[{self.name}] Changing value by delta: {delta}")
        try:
            new_value = self.value + delta
            logger.debug(f"[{self.name}] Computed new value: {new_value}")
        except TypeError as e:
            error_msg = (f"Cannot add delta of type {type(delta).__name__} to value of type "
                         f"{type(self.value).__name__}")
            logger.error(f"[{self.name}] {error_msg}")
            raise TypeError(error_msg) from e

        self._update_with_bounds(new_value)

    @abstractmethod
    def update_on_epoch(self) -> None:
        """Defines how the attribute should be updated on each epoch."""
        pass


class ConstantAgentAttribute(AgentAttribute[T]):
    """
    Represents an agent attribute with a constant value that cannot be modified after initialization.
    """

    def __setattr__(self, name: str, value: Any):
        if name == 'value' and not hasattr(self, 'value'):
            # Allow setting 'value' only during initialization
            super().__setattr__(name, value)
        else:
            error_msg = f"[{self.name}] Cannot modify 'value' of ConstantAgentAttribute."
            logger.error(error_msg)
            raise AttributeError(error_msg)

    def update_on_epoch(self) -> None:
        """Does nothing as the value remains constant across epochs."""
        logger.debug(f"[{self.name}] update_on_epoch called on ConstantAgentAttribute - no action taken.")


class MutableAgentAttribute(AgentAttribute[T]):
    """
    Represents an agent attribute whose value can be modified, optionally constrained by minimum and maximum bounds,
    and can change based on epoch updates.
    """

    epoch_change_function: Optional[Callable[[T, Optional[T], Optional[T]], T]] = Field(default=None)

    def update_on_epoch(self) -> None:
        """Updates the attribute's value based on the epoch change function, if provided."""
        if self.epoch_change_function:
            logger.debug(f"[{self.name}] Updating value based on epoch_change_function.")
            try:
                new_value = self.epoch_change_function(self.value, self.min, self.max)
                logger.debug(f"[{self.name}] Epoch change function returned new value: {new_value}")
                self._update_with_bounds(new_value)
            except Exception as e:
                logger.error(f"[{self.name}] Error during epoch update: {e}")
                raise
        else:
            logger.debug(f"[{self.name}] No epoch_change_function provided; no update performed.")


class IteratingAgentAttribute(MutableAgentAttribute[T]):
    """
    A specialized MutableAgentAttribute that increments its value by a fixed delta on each epoch.
    """

    delta: Any = Field(..., description="The fixed delta to add to the value on each epoch.")

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.epoch_change_function = self.iterate
        logger.debug(f"[{self.name}] IteratingAgentAttribute initialized with delta: {self.delta}")

    def iterate(self, current_value: T, min_value: Optional[T] = None, max_value: Optional[T] = None) -> T:
        """
        Defines the iteration behavior by adding the fixed delta to the current value.

        Args:
            current_value (T): The current value of the attribute.
            min_value (Optional[T]): The minimum allowable value.
            max_value (Optional[T]): The maximum allowable value.

        Returns:
            T: The new value after adding delta.
        """
        logger.debug(f"[{self.name}] Iterating value: current_value={current_value}, delta={self.delta}")
        try:
            new_value = current_value + self.delta
            logger.debug(f"[{self.name}] Computed new value: {new_value}")
            return new_value
        except TypeError as e:
            error_msg = (f"Cannot add delta of type {type(self.delta).__name__} to value of type "
                         f"{type(current_value).__name__}")
            logger.error(f"[{self.name}] {error_msg}")
            raise TypeError(error_msg) from e

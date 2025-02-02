from abc import ABC
from typing import Any, Callable, Optional
import logging

import numpy as np
from pydantic import BaseModel, Field, ConfigDict, model_validator

logger = logging.getLogger(__name__)

EpochChangeFunctionType = Callable[[np.ndarray, Any, Any], np.ndarray]


class AgentAttribute(BaseModel, ABC):
    """
    Represents an agent attribute with a name, description, and type.
    Optionally, it can have a minimum and maximum value (which must be of the same type as d_type)
    and a function to modify its value on an "epoch" change.
    """

    name: str = Field(..., description="Name of the agent attribute")
    description: str = Field(..., description="Description of the agent attribute")
    d_type: type = Field(..., description="Data type of the agent attribute")
    min: Optional[Any] = Field(
        default=None, description="Minimum value; must be an instance of d_type"
    )
    max: Optional[Any] = Field(
        default=None, description="Maximum value; must be an instance of d_type"
    )
    epoch_change_function: Optional[EpochChangeFunctionType] = Field(
        default=None,
        description=(
            "A function that applies an epoch change to a numpy array. "
            "It receives the array, the min and the max values."
        ),
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_min_max(cls, model: "AgentAttribute") -> "AgentAttribute":
        """
        Ensures that if provided, the min and max values are instances of d_type.
        Also checks that min is not greater than max.
        """
        d_type = model.d_type
        min_value = model.min
        max_value = model.max

        if d_type is not None:
            if min_value is not None and not isinstance(min_value, d_type):
                error_msg = (
                    f"'min' must be of type {d_type.__name__}; got {type(min_value).__name__}"
                )
                logger.debug(error_msg)
                raise ValueError(error_msg)
            if max_value is not None and not isinstance(max_value, d_type):
                error_msg = (
                    f"'max' must be of type {d_type.__name__}; got {type(max_value).__name__}"
                )
                logger.debug(error_msg)
                raise ValueError(error_msg)
            if min_value is not None and max_value is not None:
                try:
                    if min_value > max_value:
                        error_msg = "'min' must be less than or equal to 'max'"
                        logger.debug(error_msg)
                        raise ValueError(error_msg)
                except TypeError as e:
                    error_msg = f"Could not compare 'min' and 'max' values: {e}"
                    logger.debug(error_msg)
                    raise ValueError(error_msg) from e

        return model

    def __str__(self) -> str:
        # Use dict() to avoid including private attributes.
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.dict().items())
        return f"{self.__class__.__name__}({attrs})"

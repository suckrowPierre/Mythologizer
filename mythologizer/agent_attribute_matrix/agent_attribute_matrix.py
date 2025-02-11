import logging
from typing import Any, List, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from mythologizer.agent_attribute import AgentAttribute, AgentAttributeRegistry
from mythologizer.registry import Registry, KeyConfig

logger = logging.getLogger(__name__)


class AgentAttributeMatrix(BaseModel):
    """
    A matrix for storing agent attributes.
    Each row represents an agent.
    The structured dtype columns are defined by the registry.
    """
    agent_attribute_register: Registry = AgentAttributeRegistry()
    matrix: Optional[np.ndarray] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # Private attribute to store the computed dtype (a list of (name, type) tuples).
    _dtypes: List[tuple] = PrivateAttr(default=[])

    def __init__(self, **data: Any):
        """
        Extra parameters:
          - agent_attributes: a list of AgentAttribute to append to the registry.
          - attribute_values: a 2D list to convert into a structured matrix.
          - n_agents: if provided (and no matrix/attribute_values given), allocates an empty matrix.
        """
        agent_attributes: Optional[List[AgentAttribute]] = data.pop("agent_attributes", None)
        attribute_values: Optional[List[List[Any]]] = data.pop("attribute_values", None)
        n_agents: Optional[int] = data.pop("n_agents", None)

        super().__init__(**data)

        if agent_attributes:
            self.agent_attribute_register.append(agent_attributes)
            logger.debug("Appended agent attributes: %s", agent_attributes)

        # Compute dtypes for the structured numpy array based on the registry.
        self._dtypes = [
            (name, dt)
            for name, dt in zip(
                self.agent_attribute_register.names, self.agent_attribute_register.d_types
            )
        ]
        logger.debug("Computed dtypes: %s", self._dtypes)

        if self.matrix is not None and attribute_values is not None:
            msg = "Provided both 'attribute_values' and 'matrix'."
            logger.error(msg)
            raise ValueError(msg)

        if self.matrix is None:
            if attribute_values is not None:
                self.matrix = np.array(
                    [tuple(row) for row in attribute_values], dtype=self._dtypes
                )
                if self.matrix.ndim != 1:
                    msg = (
                        f"Matrix from attribute_values is not a 1D structured array. "
                        f"Shape: {self.matrix.shape}"
                    )
                    logger.error(msg)
                    raise ValueError(msg)
                logger.debug("Converted attribute_values to matrix with shape %s", self.matrix.shape)
            elif n_agents is not None:
                self.matrix = np.empty(n_agents, dtype=self._dtypes)
                logger.debug("Allocated matrix with %d rows", n_agents)

        self.validate_matrix()

        # Precompute the list of attributes that have an epoch change function.
        self.attributes_with_epoch_changing_function = [
            att for att in self.agent_attribute_register if att.epoch_change_function is not None
        ]

    @property
    def dtypes(self) -> List[tuple]:
        """Return the computed dtype of the matrix."""
        return self._dtypes

    def remove_row(self, index: int) -> None:
        """
        Remove a row from the matrix.
        """
        if self.matrix is None:
            raise ValueError("Matrix is not allocated")
        if not (0 <= index < self.matrix.shape[0]):
            raise IndexError("Row index out of bounds")
        self.matrix = np.delete(self.matrix, index, axis=0)
        logger.debug("Removed row %d. New matrix shape: %s", index, self.matrix.shape)

    def add_row(self, values: Union[dict, list, tuple, np.ndarray]) -> None:
        """
        Append a new row to the matrix.
        Values can be provided as a dict (keys must match attribute names),
        a list/tuple (order matching the attributes), or a numpy array.
        """
        if self.matrix is None:
            raise ValueError("Matrix is not allocated")

        field_names = [name for name, _ in self._dtypes]

        if isinstance(values, np.ndarray):
            if values.dtype == self.matrix.dtype:
                if values.ndim == 1:
                    new_row = values.reshape(1)
                elif values.ndim == 2 and values.shape[0] == 1:
                    new_row = values
                else:
                    raise ValueError("Provided numpy array must be a single-row structured array.")
            else:
                if values.ndim == 1 and values.shape[0] == len(self._dtypes):
                    new_row = np.array([tuple(values.tolist())], dtype=self._dtypes)
                else:
                    raise ValueError("Provided numpy array does not match expected shape or dtype.")
        elif isinstance(values, dict):
            try:
                new_row = np.array([tuple(values[name] for name in field_names)], dtype=self._dtypes)
            except KeyError as err:
                raise ValueError(f"Missing key in values: {err}") from err
        elif isinstance(values, (list, tuple)):
            if len(values) != len(self._dtypes):
                raise ValueError(f"Expected {len(self._dtypes)} values but got {len(values)}")
            new_row = np.array([tuple(values)], dtype=self._dtypes)
        else:
            raise TypeError("Row values must be a dict, list, tuple, or numpy.ndarray")

        self.matrix = np.append(self.matrix, new_row, axis=0)
        logger.debug("Added new row. New matrix shape: %s", self.matrix.shape)

    def apply_epoch_changing_functions(self) -> None:
        """
        Applies each attribute’s epoch change function to its corresponding column in the matrix.
        The function is expected to accept (values, min_val, max_val) and return an array of the same shape.
        If min or max are provided, the new values are clamped to those bounds.
        """
        if self.matrix is None:
            raise ValueError("Matrix is not allocated")
        for att in self.attributes_with_epoch_changing_function:
            current_values = self.matrix[att.name]
            new_values = att.epoch_change_function(current_values, att.min, att.max)
            if att.min is not None or att.max is not None:
                lower = att.min if att.min is not None else -np.inf
                upper = att.max if att.max is not None else np.inf
                new_values = np.clip(new_values, lower, upper)
            if new_values.shape != current_values.shape:
                raise ValueError(
                    f"Epoch change function for attribute '{att.name}' returned shape "
                    f"{new_values.shape} but expected {current_values.shape}"
                )
            self.matrix[att.name] = new_values
            logger.debug("Updated attribute '%s' with new epoch values.", att.name)

    def validate_cols(self) -> None:
        """
        Validates that for every attribute (i.e. each column), all values are within the specified min and max bounds.
        Throws a ValueError if any value falls outside the allowed range, indicating the row indices and values of the violations.
        """
        for name, _ in self._dtypes:
            att: Optional[AgentAttribute] = self.agent_attribute_register[name]
            if att is None:
                continue
            col = self.matrix[name]
            if att.min is not None:
                mask = col < att.min
                if np.any(mask):
                    indices = np.where(mask)[0]
                    wrong_values = col[mask]
                    raise ValueError(
                        f"Column '{name}' has values below the minimum {att.min} "
                        f"at rows {indices.tolist()} with values {wrong_values.tolist()}."
                    )
            if att.max is not None:
                mask = col > att.max
                if np.any(mask):
                    indices = np.where(mask)[0]
                    wrong_values = col[mask]
                    raise ValueError(
                        f"Column '{name}' has values above the maximum {att.max} "
                        f"at rows {indices.tolist()} with values {wrong_values.tolist()}."
                    )
        logger.debug("All columns validated successfully.")

    def validate_matrix(self) -> None:
        """
        Validates that the matrix is allocated, that its dtype matches the expected dtype,
        and that each column's values are within the specified bounds.
        """
        if self.matrix is None:
            raise ValueError("Matrix is not allocated")
        expected_dtype = np.dtype(self._dtypes)
        if self.matrix.dtype != expected_dtype:
            raise ValueError(
                f"Matrix dtype {self.matrix.dtype} does not match expected {expected_dtype}"
            )
        self.validate_cols()



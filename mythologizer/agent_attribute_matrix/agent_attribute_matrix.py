import logging
from typing import Any, List, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from mythologizer.agent_attribute import AgentAttribute
from mythologizer.registry import Registry, KeyConfig

logger = logging.getLogger(__name__)


class AgentAttributeMatrix(BaseModel):
    """
    A matrix for storing agent attributes.
    - Each row represents an agent.
    - The structured dtype columns are defined by the registry.
    """
    # Registry of agent attributes (each attribute should have a 'name')
    agent_attribute_register: Registry[AgentAttribute] = Field(
        default_factory=lambda: Registry(
            key_configs=[KeyConfig(prop_name="names", attr_name="name", expected_type=str)]
        )
    )
    # A structured numpy array where each row corresponds to an agent.
    matrix: Optional[np.ndarray] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # Private attribute to store the computed dtype for the numpy array.
    _dtypes: List[tuple] = PrivateAttr(default=[])

    def __init__(self, **data: Any):
        """
        Accepts two extra keyword arguments:
          - agent_attributes: a list of AgentAttribute to be appended to the registry.
          - n_agents: if provided and no attribute_agent_matrix was passed,
                     a new matrix is allocated with n_agents rows.
        """
        # Extract extra initialization parameters (they are not part of the model's public fields)
        agent_attributes: Optional[List[AgentAttribute]] = data.pop("agent_attributes", None)
        n_agents: Optional[int] = data.pop("n_agents", None)

        logger.debug(
            "Initializing AgentAttributeMatrix with agent_attributes=%s, n_agents=%s",
            agent_attributes, n_agents
        )
        super().__init__(**data)

        if agent_attributes:
            self.agent_attribute_register.append(agent_attributes)
            logger.debug("Appended agent attributes: %s", agent_attributes)

        # Compute and store the dtype for the numpy matrix based on the registry.
        self._dtypes = [
            (name, dt)
            for name, dt in zip(
                self.agent_attribute_register.names,
                self.agent_attribute_register.d_types
            )
        ]
        logger.debug("Computed dtypes: %s", self._dtypes)

        # Allocate the matrix if needed.
        if self.matrix is None and n_agents is not None:
            logger.debug("Allocating attribute_agent_matrix with %d rows", n_agents)
            self.matrix = np.empty(n_agents, dtype=self._dtypes)
            logger.debug("Allocated attribute_agent_matrix with shape %s", self.matrix.shape)

    @property
    def dtypes(self) -> List[tuple]:
        """Return the computed dtype of the matrix."""
        return self._dtypes

    def remove_row(self, index: int) -> None:
        """
        Remove a row from the attribute_agent_matrix.

        Parameters:
            index (int): The row index to remove.

        Raises:
            ValueError: If the matrix is not allocated.
            IndexError: If the index is out of bounds.
        """
        logger.debug("Attempting to remove row %d from attribute_agent_matrix", index)
        if self.matrix is None:
            logger.error("attribute_agent_matrix is not allocated")
            raise ValueError("attribute_agent_matrix is not allocated")
        if not (0 <= index < self.matrix.shape[0]):
            logger.error("Row index %d out of bounds", index)
            raise IndexError("Row index out of bounds")
        self.matrix = np.delete(self.matrix, index, axis=0)
        logger.debug("Successfully removed row %d. New shape: %s", index, self.matrix.shape)

    def add_row(self, values: Union[dict, list, tuple, np.ndarray]) -> None:
        """
        Append a new row to the matrix.

        Parameters:
            values (dict, list, tuple, or np.ndarray): The new row values.
                - If a dict is provided, keys must correspond to attribute names.
                - If a list/tuple is provided, its elements must match the order of attributes.
                - If a numpy array is provided, it should either be a structured array matching the matrix dtype
                  or a 1D array with the same number of elements as attributes.

        Raises:
            ValueError: If the matrix is not allocated or if the provided values do not match expectations.
            TypeError: If the input type is not supported.
        """

        logger.debug("Attempting to add a new row with values: %s", values)
        if self.matrix is None:
            logger.error("Matrix is not allocated")
            raise ValueError("Matrix is not allocated")

        field_names = [name for name, _ in self._dtypes]

        # Handle input if it's a NumPy array
        if isinstance(values, np.ndarray):
            if values.dtype == self.matrix.dtype:
                if values.ndim == 1:
                    new_row = values.reshape(1)
                elif values.ndim == 2 and values.shape[0] == 1:
                    new_row = values
                else:
                    logger.error("Provided numpy array must be a single row structured array.")
                    raise ValueError("Provided numpy array must be a single row structured array.")
            else:
                # If not a structured array, try converting a plain 1D array.
                if values.ndim == 1 and values.shape[0] == len(self._dtypes):
                    new_row_tuple = tuple(values.tolist())
                    new_row = np.array([new_row_tuple], dtype=self._dtypes)
                else:
                    logger.error("Provided numpy array does not match the expected shape or dtype.")
                    raise ValueError("Provided numpy array does not match the expected shape or dtype.")

        # Handle dict input
        elif isinstance(values, dict):
            try:
                new_row_tuple = tuple(values[name] for name in field_names)
            except KeyError as err:
                logger.error("Missing key in values: %s", err)
                raise ValueError(f"Missing key in values: {err}") from err
            new_row = np.array([new_row_tuple], dtype=self._dtypes)

        # Handle list/tuple input
        elif isinstance(values, (list, tuple)):
            if len(values) != len(self._dtypes):
                logger.error("Expected %d values but got %d", len(self._dtypes), len(values))
                raise ValueError(f"Expected {len(self._dtypes)} values but got {len(values)}")
            new_row_tuple = tuple(values)
            new_row = np.array([new_row_tuple], dtype=self._dtypes)

        else:
            logger.error("Unsupported type for row values: %s", type(values))
            raise TypeError("Row values must be a dict, list, tuple, or numpy.ndarray")

        logger.debug("New row structured array: %s", new_row)

        # Append the new row to the matrix.
        self.matrix = np.append(self.matrix, new_row, axis=0)
        logger.debug("Successfully added new row. New matrix shape: %s", self.matrix.shape)


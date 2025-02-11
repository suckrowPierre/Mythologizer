import logging
from typing import List, Union, Iterator, TypeVar, Generic, Optional, Any
import uuid
from pydantic import BaseModel, Field, UUID4, StrictInt
import numpy as np

from .agent_attribute import AgentAttribute
from mythologizer.registry import Registry, KeyConfig

logger = logging.getLogger(__name__)


class AgentAttributeRegistry(Registry[AgentAttribute]):
    """
    A specialized registry that always uses AgentAttribute as record type.
    Inherits from `Registry[AgentAttribute]`.
    """

    def __init__(self, records: Optional[List[AgentAttribute]] = None, **data: Any):
        if records is None:
            records = []
        key_configs = [
            KeyConfig(prop_name="names", attr_name="name", expected_type=str),
        ]
        # pass those to super
        super().__init__(key_configs=key_configs, records=records, **data)
        logger.debug("AgentAttributeRegistry initialized.")

    def create_values_dict(self, values: np.ndarray) -> List[dict]:
        if len(values) != len(self):
            raise ValueError("given values are not the same length as AgentAttributeRegistry")
        result = {}
        for i, agent_attribute in enumerate(self):
            result[agent_attribute.name] = values[i]
        return result

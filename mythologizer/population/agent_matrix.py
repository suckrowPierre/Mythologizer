import numpy as np
from uuid import UUID
from typing import List, Optional, Any, Dict, Callable, Union
from pydantic import BaseModel, Field, ConfigDict

from mythologizer.agent_attribute import AgentAttribute
from mythologizer.agent import Agent
from mythologizer.registry import Registry, KeyConfig
import logging

logger = logging.getLogger(__name__)


class Population(BaseModel):
    # agents are rows
    # attributes are cols

    agent_attribute_register: Registry[AgentAttribute] = Field(
        default_factory=lambda: Registry(
            key_configs=[KeyConfig(prop_name="names", attr_name="name", expected_type=str)]
        )
    )
    agent_register: Registry[Agent] = Field(
        default_factory=lambda: Registry(
            key_configs=[KeyConfig(prop_name="ids", attr_name="id", expected_type=UUID)]
        )
    ) # TODO THIS ACTUALL NEEDS TO BE A TREE
    attribute_agent_matrix: np.ndarray = Field(default_factory=lambda: np.empty(0))

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, agent_attributes: List[AgentAttribute] = None, agents: List[Agent] = None, **data):
        super().__init__(**data)
        if agent_attributes:
            self.agent_attribute_register.append(agent_attributes)
        if agents:
            self.agent_register.append(agents)

        dtype = [(name, type) for name, type in zip(self.agent_attribute_register.names, self.agent_attribute_register.d_types)]
        n_agents = len(self.agent_register)
        self.attribute_agent_matrix = np.empty(n_agents, dtype=dtype)


    def append_agents(self):
        pass

    def remove_agents(self):
        pass

    def genocide(self) -> None:
        """
        Eliminate all agents from the population.

        Logs the initiation and completion of the genocide event.
        """
        logger.info(f"Initiating genocide. Current population: {len(self.agents)}")
        # TODO delte agents in matrix
        self.agent_register.clear()
        logger.info("Genocide committed. Population is now empty.")
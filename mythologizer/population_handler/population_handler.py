import numpy as np
import logging
from uuid import UUID
from typing import List, Optional, Any, Union
from pydantic import BaseModel, Field, ConfigDict

from mythologizer.population import Population
from mythologizer.agent_attribute_matrix import AgentAttributeMatrix
from mythologizer.agent import Agent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PopulationHandler(BaseModel):
    population: Optional[Population] = None
    alive_agents_uuids: List[UUID] = Field(default_factory=list)
    agent_attribute_matrix: Optional[AgentAttributeMatrix] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
            self,
            agent_attributes: Optional[List[Any]] = None,
            agents: Optional[List[Agent]] = None,
            **data: Any
    ):
        """
        Optionally accepts `agent_attributes` to initialize the AgentAttributeMatrix and
        `agents` to initialize the Population.

        Either an agent_attribute_matrix must be provided via data or a list of agent_attributes
        must be provided.
        """
        super().__init__(**data)
        logger.debug("Initializing PopulationHandler with data: %s", data)

        if self.population is not None:
            self.population.reset_indices()

        # Initialize population with provided agents if necessary.
        if agents is not None:
            # If population does not have agents, reinitialize it.
            if self.population is None or self.population.agents is None or self.population.agents == {}:
                self.population = Population(agents=agents)
                logger.debug("Created new Population with agents: %s", agents)
            else:
                msg = "Both agents and an existing population provided"
                logger.error(msg)
                raise ValueError(msg)

        # Set alive_agents_uuids from the population if not already set.
        if not self.alive_agents_uuids:
            self.alive_agents_uuids = self.population.alive_uuids
            logger.debug("Set alive_agents_uuids from population: %s", self.alive_agents_uuids)

        # Ensure that the number of alive agents matches the population size.
        if len(self.alive_agents_uuids) != len(self.population.alive_agents):
            msg = "Mismatch between alive_agents_uuids length and population.alive_agents size"
            logger.error(msg)
            raise ValueError(msg)

        for i, uuid in enumerate(self.alive_agents_uuids):
            self.population[uuid].index = i
            logger.debug(f"Set agents index with uuid {uuid} to {i}")

        if agent_attributes is not None:
            if self.agent_attribute_matrix is None:
                self.agent_attribute_matrix = AgentAttributeMatrix(agent_attributes=agent_attributes,
                                                                   n_agents=len(self.alive_agents_uuids))
                logger.debug("Created new AgentAttributeMatrix with attributes: %s", agent_attributes)
            else:
                msg = "Both agent_attributes and agent_attribute_matrix provided"
                logger.error(msg)
                raise ValueError(msg)
        elif self.agent_attribute_matrix is None:
            msg = ("Could not initialize AgentAttributeMatrix. Either an agent_attributes "
                   "list or a pre-built agent_attribute_matrix must be provided.")
            logger.error(msg)
            raise ValueError(msg)

        logger.debug("PopulationHandler initialized successfully.")


    def get_index_from_agent_uuid(self, agent_id: UUID) -> int:
        """
        Return the index of the agent given its UUID.
        """
        agent = self.population[agent_id]
        if agent is None:
            msg = f"Agent with UUID {agent_id} not found in population."
            logger.error(msg)
            raise ValueError(msg) from e
        logger.debug("Found agent %s at index %d", agent_id, agent.index)
        return agent.index


    def get_agent_uuid_from_index(self, index: int) -> UUID:
        """
        Return the UUID of the agent at the given index in the alive agents list.
        """
        try:
            agent_uuid = self.alive_agents_uuids[index]
            logger.debug("Found agent UUID %s at index %d", agent_uuid, index)
            return agent_uuid
        except IndexError as e:
            msg = f"Index {index} is out of range for alive_agents_uuids."
            logger.error(msg)
            raise ValueError(msg) from e

    def add_agent(self, agent: Agent, values: Union[dict, list, tuple, np.ndarray, None] = None) -> None:
        """
        Add an agent to the population, assign its index, update the alive_agents_uuids list,
        and add a corresponding row to the attribute matrix.
        """
        index = len(self.alive_agents_uuids)
        agent.index = index
        self.alive_agents_uuids.append(agent.id)
        self.population[agent.id] = agent
        logger.debug("Added agent %s at index %d", agent.id, index)

        self.agent_attribute_matrix.add_row(values)
        logger.debug("Added row to attribute matrix with values: %s", values)

    def kill_agent(self, agent_or_id: Union[Agent, UUID]) -> None:
        """
        Kill an agent given the Agent instance or its UUID. This updates the population,
        adjusts indices for agents that follow, and removes the corresponding row from the attribute matrix.
        """
        if isinstance(agent_or_id, Agent):
            agent_id = agent_or_id.id
        elif isinstance(agent_or_id, UUID):
            agent_id = agent_or_id
        else:
            msg = f"agent_or_id must be an Agent or UUID, got {type(agent_or_id)}"
            logger.error(msg)
            raise TypeError(msg)

        logger.debug("Attempting to kill agent with UUID: %s", agent_id)
        index = self.get_index_from_agent_uuid(agent_id)

        self.population.kill_agents(agent_id)
        logger.debug("Killed agent %s in population.", agent_id)

        for subsequent_uuid in self.alive_agents_uuids[index + 1:]:
            agent_to_update = self.population[subsequent_uuid]
            agent_to_update.index -= 1
            logger.debug("Adjusted index for agent %s to %d", subsequent_uuid, agent_to_update.index)

        self.alive_agents_uuids.pop(index)
        logger.debug("Removed agent UUID %s from alive_agents_uuids.", agent_id)

        self.agent_attribute_matrix.remove_row(index)
        logger.debug("Removed row %d from attribute matrix.", index)

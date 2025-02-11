import logging
from uuid import UUID
from typing import List, Optional, Any, Union

from pydantic import BaseModel, Field, ConfigDict

from mythologizer.population import Population
from mythologizer.agent_attribute_matrix import AgentAttributeMatrix
from mythologizer.agent import Agent
from mythologizer.culture import CultureRegistry, Culture
from mythologizer.agent_attribute import AgentAttribute

logger = logging.getLogger(__name__)


class AgentLifecycleManager(BaseModel):
    population: Optional[Population] = None
    alive_agents_uuids: List[UUID] = Field(default_factory=list)
    agent_attribute_matrix: Optional[AgentAttributeMatrix] = None
    culture_registry: Optional[CultureRegistry] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        agent_attributes: Optional[List[AgentAttribute]] = None,
        agents: Optional[List[Agent]] = None,
        cultures: Optional[List[Culture]] = None,
        attribute_values: Optional[List[any]] = None,
        **data: Any
    ):
        """
        # TODO: values of agents as a List in the params
        Optionally accepts `agent_attributes` to initialize the AgentAttributeMatrix and
        `agents` to initialize the Population.

        Either an agent_attribute_matrix must be provided via data or a list of agent_attributes
        must be provided.
        """
        super().__init__(**data)
        logger.debug("Initializing AgentLifecycleManager with data: %s", data)

        # If an existing population is provided, reset its agent indices.
        if self.population is not None:
            self.population.reset_indices()

        # Initialize the population if agents are provided.
        if agents is not None:
            if (self.population is None or not self.population.agents):
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

        # Ensure the alive agents list matches the population.
        if len(self.alive_agents_uuids) != len(self.population.alive_agents):
            msg = "Mismatch between alive_agents_uuids length and population.alive_agents size"
            logger.error(msg)
            raise ValueError(msg)

        for i, agent_uuid in enumerate(self.alive_agents_uuids):
            self.population[agent_uuid].index = i
            logger.debug("Set agent %s index to %d", agent_uuid, i)

        # Initialize the agent attribute matrix.
        if agent_attributes is not None:
            if self.agent_attribute_matrix is None:
                if attribute_values is None:
                    self.agent_attribute_matrix = AgentAttributeMatrix(
                        agent_attributes=agent_attributes, n_agents=len(self.alive_agents_uuids)
                    )
                    logger.debug("Created new AgentAttributeMatrix with attributes: %s", agent_attributes)
                else:
                    self.agent_attribute_matrix = AgentAttributeMatrix(
                        agent_attributes=agent_attributes, n_agents=len(self.alive_agents_uuids),
                        attribute_values=attribute_values
                    )
                    logger.debug("Created new AgentAttributeMatrix with attributes: %s and given attribute values", agent_attributes)
            else:
                msg = "Both agent_attributes and agent_attribute_matrix provided"
                logger.error(msg)
                raise ValueError(msg)
        elif self.agent_attribute_matrix is None:
            msg = ("Could not initialize AgentAttributeMatrix. Either an agent_attributes "
                   "list or a pre-built agent_attribute_matrix must be provided.")
            logger.error(msg)
            raise ValueError(msg)

        # Optionally initialize the culture register if cultures are provided.
        if cultures is not None:
            if self.culture_registry is None:
                self.culture_registry = CultureRegistry(records=cultures)
                logger.debug("Created new CultureRegistry with cultures: %s", cultures)
            else:
                # If a culture register already exists, append the new cultures.
                self.add_culture(cultures)

        logger.debug("AgentLifecycleManager initialized successfully.")

    @staticmethod
    def _extract_agent_id(agent_or_id: Union[Agent, UUID]) -> UUID:
        """Extract a UUID from an Agent or a UUID."""
        if isinstance(agent_or_id, Agent):
            return agent_or_id.id
        if isinstance(agent_or_id, UUID):
            return agent_or_id
        raise TypeError(f"agent_or_id must be an Agent or UUID, got {type(agent_or_id)}")

    @staticmethod
    def _extract_culture_ids(
        cultures: Union[Culture, UUID, List[Union[Culture, UUID]]]
    ) -> List[UUID]:
        """
        Convert a culture or a list of cultures (or UUIDs) into a list of UUIDs.
        """
        if not isinstance(cultures, list):
            cultures = [cultures]
        culture_ids = []
        for item in cultures:
            if isinstance(item, Culture):
                culture_ids.append(item.id)
            elif isinstance(item, UUID):
                culture_ids.append(item)
            else:
                raise TypeError(f"Each culture must be a Culture instance or UUID, got {type(item)}")
        return culture_ids

    def get_index_from_agent_uuid(self, agent_id: UUID) -> int:
        """
        Return the index of the agent given its UUID.
        """
        try:
            agent = self.population[agent_id]
        except KeyError:
            msg = f"Agent with UUID {agent_id} not found in population."
            logger.error(msg)
            raise ValueError(msg)
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

    def add_agent(self, agent: Agent, values: Union[dict, list, tuple, None] = None) -> None:
        """
        Add an agent to the population. Assign its index, update the alive_agents_uuids list,
        update the population, add the agent to any associated cultures, and add a corresponding
        row to the attribute matrix.
        """
        index = len(self.alive_agents_uuids)
        agent.index = index
        self.alive_agents_uuids.append(agent.id)
        self.population[agent.id] = agent
        logger.debug("Added agent %s at index %d", agent.id, index)

        if agent.culture_ids:
            for culture_id in agent.culture_ids:
                culture = self.culture_registry[culture_id]
                culture.active_member_ids.add(agent.id)
                logger.debug("Added agent %s to culture %s", agent.id, culture.name)

        self.agent_attribute_matrix.add_row(values)
        logger.debug("Added row to attribute matrix with values: %s", values)

    def kill_agent(self, agent_or_id: Union[Agent, UUID]) -> None:
        """
        Kill an agent given its instance or UUID. This updates the population,
        adjusts indices for subsequent agents, and removes the corresponding row from
        the attribute matrix.
        """
        agent_id = self._extract_agent_id(agent_or_id)
        logger.debug("Attempting to kill agent with UUID: %s", agent_id)
        index = self.get_index_from_agent_uuid(agent_id)

        self.remove_culture_from_agent(agent_or_id, list(self.population[agent_id].culture_ids))

        self.population.kill_agents(agent_id)
        logger.debug("Killed agent %s in population.", agent_id)

        # Adjust the indices of agents that follow the killed agent.
        for subsequent_uuid in self.alive_agents_uuids[index + 1:]:
            agent_to_update = self.population[subsequent_uuid]
            agent_to_update.index -= 1
            logger.debug("Adjusted index for agent %s to %d", subsequent_uuid, agent_to_update.index)

        self.alive_agents_uuids.pop(index)
        logger.debug("Removed agent UUID %s from alive_agents_uuids.", agent_id)

        self.agent_attribute_matrix.remove_row(index)
        logger.debug("Removed row %d from attribute matrix.", index)


    def add_culture(self, culture: Union[Culture, List[Culture]]) -> None:
        """
        Add one or more cultures to the culture register and update each culture’s active members
        in the population.
        """
        if not isinstance(culture, list):
            cultures = [culture]
        else:
            cultures = culture

        self.culture_registry.append(cultures)
        for c in cultures:
            for agent_id in c.active_member_ids:
                self.population[agent_id].culture_ids.add(c.id)
                logger.debug("Added culture %s to agent %s", c.name, agent_id)

    def delete_culture(self, culture: Union[UUID, Culture, List[Union[Culture, UUID]]]) -> None:
        """
        Delete one or more cultures from the culture register and remove the culture
        membership from all associated agents.
        """
        culture_ids = self._extract_culture_ids(culture)
        for cid in culture_ids:
            try:
                culture = self.culture_registry[cid]
            except KeyError:
                msg = f"Culture with id {cid} not found in the culture registry."
                logger.error(msg)
                raise ValueError(msg)
            # Remove the culture id from each active agent.
            for agent_id in list(culture.active_member_ids):
                self.population[agent_id].culture_ids.discard(cid)
                logger.debug("Removed culture %s from agent %s", cid, agent_id)
            del self.culture_registry[cid]
            logger.debug("Deleted culture %s from culture registry", cid)

    def add_culture_to_agent(
        self,
        agent_or_id: Union[Agent, UUID],
        cultures: Union[Culture, UUID, List[Union[Culture, UUID]]]
    ) -> None:
        """
        Add one or more cultures to a given agent. This updates both the agent’s culture_ids
        and the culture registry’s active members.
        """
        agent_id = self._extract_agent_id(agent_or_id)
        culture_ids = self._extract_culture_ids(cultures)

        try:
            agent = self.population[agent_id]
        except KeyError:
            msg = f"Agent with id {agent_id} not found in population."
            logger.error(msg)
            raise ValueError(msg)

        agent.culture_ids.update(culture_ids)
        logger.debug("Added cultures %s to agent %s", culture_ids, agent_id)

        for cid in culture_ids:
            try:
                self.culture_registry[cid].active_member_ids.add(agent_id)
                logger.debug("Added agent %s to culture %s", agent_id, cid)
            except KeyError:
                msg = f"Culture {cid} is not in culture registry."
                logger.error(msg)
                raise ValueError(msg)

    def remove_culture_from_agent(
        self,
        agent_or_id: Union[Agent, UUID],
        cultures: Union[Culture, UUID, List[Union[Culture, UUID]]]
    ) -> None:
        """
        Remove one or more cultures from a given agent. This updates the agent’s culture_ids,
        removes the agent from the culture’s active members, and adds it to the culture’s past members.
        """
        agent_id = self._extract_agent_id(agent_or_id)
        culture_ids = self._extract_culture_ids(cultures)

        try:
            agent = self.population[agent_id]
        except KeyError:
            msg = f"Agent with id {agent_id} not found in population."
            logger.error(msg)
            raise ValueError(msg)

        agent.culture_ids.difference_update(culture_ids)
        logger.debug("Removed cultures %s from agent %s", culture_ids, agent_id)

        for cid in culture_ids:
            try:
                culture = self.culture_registry[cid]
                culture.active_member_ids.discard(agent_id)
                culture.past_member_ids.add(agent_id)
                logger.debug("Moved agent %s to past members of culture %s", agent_id, cid)
            except KeyError:
                msg = f"Culture {cid} is not in culture registry."
                logger.error(msg)
                raise ValueError(msg)

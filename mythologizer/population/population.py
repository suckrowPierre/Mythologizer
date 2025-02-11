import logging
from typing import List, Union, Optional, Dict
from uuid import UUID
from pydantic import BaseModel, Field, validate_call, UUID4
from mythologizer.agent import Agent

logger = logging.getLogger(__name__)


class Population(BaseModel):
    """
    A collection of agents, separating living and dead ones.

    Agents are stored in dictionaries keyed by their UUID.
    """
    alive_agents: Dict[UUID4, Agent] = Field(default_factory=dict)
    dead_agents: Dict[UUID4, Agent] = Field(default_factory=dict)

    def __init__(self, agents: Optional[List[Agent]] = None, **kwargs):
        """
        Optionally initialize the population with a list of agents.
        All supplied agents will be added as alive.
        """
        super().__init__(**kwargs)
        if agents:
            self.append(agents)

    @property
    def alive_uuids(self) -> List[UUID4]:
        return [agent.id for agent in self.alive_agents.values()]


    @property
    def agents(self) -> Dict[UUID4, Agent]:
        """
        Returns all agents (both alive and dead) as a single dictionary.
        """
        # Using dictionary unpacking for compatibility with older Python versions.
        return {**self.alive_agents, **self.dead_agents}
        # Alternatively, for Python 3.9+ you can use:
        # return self.alive_agents | self.dead_agents

    @validate_call
    def append(self, agent: Union[Agent, List[Agent]]) -> None:
        """
        Adds an agent—or a list of agents—to the alive agents.

        If an agent already exists (by id), it is overwritten.
        """
        if isinstance(agent, list):
            for a in agent:
                self.append(a)
            return

        if agent.id in self.alive_agents:
            logger.warning("Agent with id %s already exists in alive_agents; overwriting.", agent.id)
        self.alive_agents[agent.id] = agent
        logger.info("Appended agent with id %s", agent.id)

    @validate_call
    def kill_agents(self, agent: Union[Agent, UUID, List[Union[Agent, UUID]]]) -> None:
        """
        Moves an agent—or a list of agents—from alive_agents to dead_agents.

        If an agent isn’t found among the alive agents, a warning is logged.
        """
        if isinstance(agent, list):
            for a in agent:
                self.kill_agents(a)
            return

        # Instead of checking against UUID4 (a subscripted generic), we simply check for UUID.
        if isinstance(agent, UUID):
            agent_id = agent
        else:
            agent_id = agent.id

        if agent_id in self.alive_agents:
            self.dead_agents[agent_id] = self.alive_agents.pop(agent_id)
            logger.info("Killed agent with id %s", agent_id)
        else:
            logger.warning("Attempted to kill agent with id %s, but agent not found in alive_agents.", agent_id)

    @validate_call
    def is_dead(self, agent_id: Union[UUID4, List[UUID4]]) -> Union[bool, List[bool]]:
        """
        Checks if the given agent id (or list of ids) corresponds to a dead agent.

        Returns a boolean (or list of booleans) indicating whether the agent(s) is/are dead.
        """
        if isinstance(agent_id, list):
            return [self.is_dead(single_id) for single_id in agent_id]
        return agent_id in self.dead_agents

    def __len__(self) -> int:
        """
        Returns the total number of agents (both alive and dead).
        """
        return len(self.alive_agents) + len(self.dead_agents)

    def __getitem__(self, key: UUID4) -> Agent:
        """
        Retrieves an agent by id, searching among alive and dead agents.
        """
        if key in self.alive_agents:
            return self.alive_agents[key]
        if key in self.dead_agents:
            return self.dead_agents[key]
        raise KeyError(f"Agent with id {key} not found.")

    def __setitem__(self, key: UUID4, value: Agent) -> None:
        """
        Supports assignment using square brackets, e.g.:

            population[agent.id] = agent

        If the key already exists in dead_agents, the dead agent is updated;
        otherwise the agent is (re)stored in alive_agents.
        """
        if key in self.dead_agents:
            self.dead_agents[key] = value
            logger.info("Updated dead agent with id %s", key)
        else:
            self.alive_agents[key] = value
            logger.info("Set alive agent with id %s", key)

    def reset_indices(self) -> None:
        for agent in self.agents.value():
            logger.debug(f"Set agent {agent.id} index to None")
            agent.index = None

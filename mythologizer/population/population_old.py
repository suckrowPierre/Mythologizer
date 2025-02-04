from typing import List, Optional, Tuple, Callable
import logging
from pydantic import BaseModel, Field, StrictInt, validate_call
from mythologizer import BaseAgent, RandomAgent, CultureRegister, RNG
import numpy as np

logger = logging.getLogger(__name__)







class Population(BaseModel):
    """
    Represents a population of agents with associated cultural attributes.

    Attributes:
        agents (List[BaseAgent]): A list of agents within the population.
    """
    rng: RNG
    agents: List[BaseAgent] = Field(default_factory=list)

    def __init__(self, **data):
        """
        Initialize a Population instance and log its creation.

        Args:
            **data: Arbitrary keyword arguments corresponding to the model fields.
        """
        super().__init__(**data)
        logger.debug(f"Population class created with {len(self.agents)} agents.")

    def genocide(self) -> None:
        """
        Eliminate all agents from the population.

        Logs the initiation and completion of the genocide event.
        """
        logger.info(f"Initiating genocide. Current population: {len(self.agents)}")
        self.agents.clear()
        logger.info("Genocide committed. Population is now empty.")

    @validate_call
    def random_population(
            self,
            n_agents: StrictInt = Field(..., ge=1, description="Must be a positive integer >= 1"),
            culture_register: CultureRegister = Field(..., description="Register must have length >= 1"),
            culture_distribution_function: Callable = get_cultures_random_with_overlap
    ) -> None:
        """
        Populate the population with a random distribution of agents across cultures.

        Args: n_agents (StrictInt): Number of agents to populate. Must be >= 1.
        culture_register (CultureRegister): Register containing available cultures. Must have at least one culture.
        culture_distribution_function: Function to return a list of tuples the length of n_agents with the tuples
        indicating which cultures an agent gets

        Raises:
            ValueError: If the culture_register contains fewer than one culture.
        """
        n_cultures = len(culture_register)

        if n_cultures < 1:
            logger.error("Culture register has insufficient cultures.")
            raise ValueError("culture_register must have length >= 1")

        culture_names = ' '.join(culture_register.get_names())
        logger.info(
            f"Randomly populating {n_agents} agents across cultures: {culture_names}"
        )
        logger.debug(
            f"Using {culture_distribution_function.__name__} for dictating cultures to agent"
        )

        culture_tuples = culture_distribution_function(n_agents=n_agents, n_cultures=n_cultures, rng=self.rng)

        for culture_tuple in culture_tuples:
            cultures = culture_register[list[culture_tuple]]
            new_agent = RandomAgent(cultures=cultures, rng=self.rng)
            self.agents.append(new_agent)
            logger.debug(f"Added new agent with cultures '{', '.join([culture.name for culture in cultures])}'")

        logger.info("Completed random population of agents.")

    def update_to_affected_culture(self, affected_cultures: List[str]) -> None:
        """
        Update agents' attributes based on the affected cultures.

        Args:
            affected_cultures (List[str]): A list of culture names that are affected.

        Notes:
            This method currently contains a placeholder for the update logic.
        """
        logger.info(f"Updating agents to affected cultures: {affected_cultures}")
        for agent in self.agents:
            # Placeholder for updating agent attributes based on affected cultures
            logger.debug(f"Processing agent {agent.id} for culture updates.")
        logger.info("Completed updating agents to affected cultures.")

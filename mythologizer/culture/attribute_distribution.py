from typing import List, Dict, Any
import logging

import numpy as np
from pydantic import BaseModel, root_validator, Field

from mythologizer.random_number_generator import ProbabilityDistribution

logger = logging.getLogger(__name__)


class AttributeDistribution(BaseModel):
    name: str
    distribution: ProbabilityDistribution
    parameters: Dict[str, Any]


class AttributesDistributions(BaseModel):
    attributes_distributions: List[AttributeDistribution] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.attributes_distributions)

    @property
    def keys(self) -> List[str]:
        """Names of all probability distributions."""
        return [attr.name for attr in self.attributes_distributions]

    def __getitem__(self, key: str) -> ProbabilityDistribution:
        """Retrieve a distribution by its name."""
        for attr in self.attributes_distributions:
            if attr.name == key:
                return attr.distribution
        error_msg = f"Unsupported agent attribute key: '{key}'. Supported keys: {self.keys}."
        logger.error(error_msg)
        raise KeyError(error_msg)

    def sample(self, n_agents: int) -> np.ndarray:
        """Sample values for all attributes for a given number of agents."""
        samples = [
            attr.distribution.sample(parameters=attr.parameters, size=n_agents)
            for attr in self.attributes_distributions
        ]
        return np.vstack(samples)

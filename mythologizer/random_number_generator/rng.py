import numpy as np

from typing import List, Optional, Tuple, Callable, Dict, Any
import logging
from pydantic import BaseModel, Field, StrictInt, validate_call

from .distributions import ProbabilityDistributionMap, ProbabilityDistribution

logger = logging.getLogger(__name__)


class RandomNumberGenerator:
    def __init__(self, seed: Optional[int] = None):
        self.numpy_rng = np.random.default_rng(seed)
        self.distributions = ProbabilityDistributionMap(distributions=[
            ProbabilityDistribution(
                name="beta",
                mapping=self.numpy_rng.beta,
                parameters={
                    "a": lambda x: isinstance(x, (int, float)) and x > 0,
                    "b": lambda x: isinstance(x, (int, float)) and x > 0,
                }
            ),
            ProbabilityDistribution(
                name="binomial",
                mapping=self.numpy_rng.binomial,
                parameters={
                    "n": lambda x: isinstance(x, int) and x >= 0,
                    "p": lambda x: isinstance(x, float) and 0 <= x <= 1,
                }
            ),
            ProbabilityDistribution(
                name="chisquare",
                mapping=self.numpy_rng.chisquare,
                parameters={
                    "df": lambda x: isinstance(x, (int, float)) and x > 0,
                }
            )
        ])

    def random_float(self, min_value=None, max_value=None, size=None):
        if min_value is None and max_value is None:
            return self.numpy_rng.random(size)  # Returns a random float between 0 and 1
        return min_value + self.numpy_rng.random(size) * (max_value - min_value)

    def rand_np_int(self, min_value=0, max_value=None, size=None):
        return self.numpy_rng.integers(low=min_value, high=max_value, size=size, endpoint=True)

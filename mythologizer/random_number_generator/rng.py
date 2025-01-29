import numpy as np

from typing import List, Optional, Tuple, Callable, Dict, Any
import logging
from pydantic import BaseModel, Field, StrictInt, validate_call

class RandomNumberGenerator:
    def __init__(self, seed=None):
        self.numpy_rng = np.random.default_rng(seed)
        self.dist_map = {
            "beta": {
                "mapping": self.numpy_rng.beta,
                "parameters": {
                    "a": lambda x: x > 0,
                    "b": lambda x: x > 0,
                },
            },
            "binomial": {
                "mapping": self.numpy_rng.binomial,
                "parameters": {
                    "n": lambda x: x >= 0,
                    "p": lambda x: 0 <= x <= 1,
                },
            },
            "chisquare": {
                "mapping": self.numpy_rng.chisquare,
                "parameters": {
                    "df": lambda x: x > 0,
                },
            }
        }

    def random_float(self, min_value=None, max_value=None, size=None):
        if min_value is None and max_value is None:
            return self.numpy_rng.random(size)  # Returns a random float between 0 and 1
        return min_value + self.numpy_rng.random(size) * (max_value - min_value)

    def rand_np_int(self, min_value=0, max_value=None, size=None):
        return self.numpy_rng.integers(low=min_value, high=max_value, size=size, endpoint=True)

    def sample_from_distribution(self, key: str, parameters: Dict[str, Any], size: Optional[int] = None) -> Any:
        """
        Samples values from the specified distribution with given parameters.

        Args:
            key (str): The distribution key (e.g., "beta", "binomial", "chisquare").
            parameters (Dict[str, Any]): A dictionary of parameters required for the distribution.
            size (Optional[int]): The number of samples to generate.

        Returns:
            Any: The sampled values from the distribution.

        Raises:
            ValueError: If the distribution key is unsupported or parameters are invalid.
        """
        if key not in self.dist_map:
            raise ValueError(f"Unsupported distribution key: '{key}'. Supported keys are: {list(self.dist_map.keys())}")

        dist_info = self.dist_map[key]
        required_params = dist_info["parameters"]

        # Validate provided parameters
        for param, validator in required_params.items():
            if param not in parameters:
                raise ValueError(f"Missing required parameter '{param}' for distribution '{key}'.")
            if not validator(parameters[param]):
                raise ValueError(f"Invalid value for parameter '{param}': {parameters[param]}.")

        # Check for unexpected parameters
        for param in parameters:
            if param not in required_params:
                raise ValueError(f"Unexpected parameter '{param}' for distribution '{key}'.")

        # Sample from the distribution
        try:
            sample = dist_info["mapping"](**parameters, size=size)
        except TypeError as e:
            raise ValueError(f"Error sampling from distribution '{key}': {e}")

        return sample

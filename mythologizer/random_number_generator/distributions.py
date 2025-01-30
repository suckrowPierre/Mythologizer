import numpy as np

from typing import List, Optional, Tuple, Callable, Dict, Any
import logging
from pydantic import BaseModel, Field, StrictInt, validate_call
import inspect
import json

logger = logging.getLogger(__name__)


class ProbabilityDistribution(BaseModel):
    name: str
    mapping: Callable[..., Any]
    parameters: Dict[str, Callable[[Any], bool]]

    def __str__(self):
        """
        Returns a JSON-formatted string representation of the ProbabilityDistribution instance.
        Function objects are represented as their source code strings if available,
        otherwise as a descriptive placeholder.
        """
        param_info = {}
        for param_name, func in self.parameters.items():
            try:
                # Attempt to retrieve the source code of the function
                source_lines, _ = inspect.getsourcelines(func)
                source = ''.join(source_lines).strip()

                # If it's a lambda, attempt to extract the lambda expression
                if func.__name__ == "<lambda>":
                    # Find the lambda expression in the source code
                    lambda_index = source.find("lambda")
                    if lambda_index != -1:
                        source = source[lambda_index:]
            except (OSError, TypeError):
                # Fallback if source code can't be retrieved
                if func.__name__ == "<lambda>":
                    source = "<lambda>"
                else:
                    source = f"<{func.__name__}>"

            param_info[param_name] = source

            output = {
                "name": self.name,
                "parameters": param_info
            }

        # Serialize the dictionary to a JSON-formatted string with indentation
        return json.dumps(output)

    def sample(self, parameters: Dict[str, Any], size: Optional[int] = None) -> Any:
        """
        Samples from the distribution using the provided parameters.

        Args:
            parameters (Dict[str, Any]): Parameters required for the distribution.
            size (Optional[int]): Number of samples to generate.

        Returns:
            Any: Sampled values.

        Raises:
            ValueError: If parameters are missing, invalid, or unexpected.
        """
        # Validate required parameters
        for param, validator_func in self.parameters.items():
            if param not in parameters:
                error_msg = f"Missing required parameter '{param}' for distribution '{self.name}'."
                logger.error(error_msg)
                raise ValueError(error_msg)
            if not validator_func(parameters[param]):
                error_msg = f"Invalid value for parameter '{param}': {parameters[param]}."
                logger.error(error_msg)
                raise ValueError(error_msg)

        # Check for unexpected parameters
        unexpected_params = set(parameters.keys()) - set(self.parameters.keys())
        if unexpected_params:
            error_msg = f"Unexpected parameters for distribution '{self.name}': {unexpected_params}."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Sample from the distribution
        try:
            sample = self.mapping(size=size, **parameters)
            logger.debug(f"Sampled {size if size else 'default'} values from '{self.name}' distribution.")
            return sample
        except TypeError as e:
            error_msg = f"Error sampling from distribution '{self.name}': {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e


class ProbabilityDistributionMap(BaseModel):
    distributions: List[ProbabilityDistribution] = Field(default=[])

    @property
    def keys(self) -> List[str]:
        """Retrieve the names of all probability distributions."""
        return [dist.name for dist in self.distributions]

    def get_dict_list(self):
        return [dict(dist) for dist in self.distributions]

    def __getitem__(self, key: str) -> ProbabilityDistribution:
        """Retrieve a distribution by its name."""
        for dist in self.distributions:
            if dist.name == key:
                return dist
        error_msg = f"Unsupported distribution key: '{key}'. Supported keys are: {self.keys}."
        logger.error(error_msg)
        raise KeyError(error_msg)

    def __len__(self):
        return len(self.distributions)

    def __str__(self):
        return str([str(dist) for dist in self.distributions])

from typing import Any
import logging
import uuid
from pydantic import BaseModel, Field, UUID4

logger = logging.getLogger(__name__)

from .attribute_distribution import AttributesDistributions

class Culture(BaseModel):
    """
    Represents a culture with a unique identifier, name, and description.
    """
    name: str
    description: str
    id: UUID4 = Field(default_factory=uuid.uuid4)
    attribute_distributions: AttributesDistributions = AttributesDistributions()

    def __init__(self, **data: Any) -> None:
        """
        Initialize a Culture instance and log its creation.

        Args:
            **data: Arbitrary keyword arguments corresponding to the model fields.
        """
        super().__init__(**data)
        logger.debug(
            f"Culture '{self.name}' created with id {self.id} and description '{self.description}'."
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.model_fields:
            old_value = getattr(self, name, None)
            super().__setattr__(name, value)
            logger.debug(
                f"Culture '{self.name}': Field '{name}' changed from '{old_value}' to '{value}'."
            )
        else:
            super().__setattr__(name, value)

    def __str__(self):
        return f"Culture '{self.name}': {self.description}"

    def __repr__(self):
        return f"'Culture {self.name}', id: {self.id}, description: {self.description}"

    def get_attributes_prob_distribution_from_name_and_description(self, agent_attributes, probability_functions, transform_function):
        # TODO: self.attribute_distribution = transform_function(agent_attributes, name, description)
        pass

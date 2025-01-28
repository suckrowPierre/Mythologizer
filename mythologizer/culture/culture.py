from typing import Any
import logging
import uuid
from pydantic import BaseModel, Field, UUID4

logger = logging.getLogger(__name__)


class Culture(BaseModel):
    """
    Represents a culture with a unique identifier, name, and description.
    """
    name: str
    description: str
    id: UUID4 = Field(default_factory=uuid.uuid4)

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

from typing import Set
import uuid
from pydantic import BaseModel, Field


class Myth(BaseModel):
    """
    Represents a myth with its narrative, associated themes, and a unique identifier.
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    current_myth: str
    mythemes: Set[str]
    retention: float = 1.0
    # original_myth_id: uuid.UUID TODO: currently dont know what to do with this.

    def __eq__(self, other: object) -> bool:
        """
        Override equality to compare only the unique `id` field.
        """
        if not isinstance(other, Myth):
            return NotImplemented
        return self.id == other.id

    def same_old_myth(self, other: 'Myth') -> bool:
        """
        Check if two myths originated from the same original myth.

        Args:
            other (Myth): Another myth instance to compare against.

        Returns:
            bool: True if both myths share the same original_myth_id; False otherwise.
        """
        return self.original_myth_id == other.original_myth_id

    def compare_mythemes(self, other: 'Myth') -> float:
        """
        Compare the mythemes of this myth with another myth.

        Returns the ratio of the size of the intersection to the size of the union
        of the mythemes sets. If both sets are empty, returns 0.0.

        Args:
            other (Myth): Another myth instance whose mythemes will be compared.

        Returns:
            float: The similarity ratio of the mythemes sets.
        """
        # TODO: make use of embeddings

        intersection = self.mythemes & other.mythemes
        union = self.mythemes | other.mythemes

        if not union:
            return 0.0

        return len(intersection) / len(union)

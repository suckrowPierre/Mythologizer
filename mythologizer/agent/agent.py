from __future__ import annotations
import uuid
import logging
from typing import List, Optional, Any, Tuple
from pydantic import BaseModel, Field, UUID4

logger = logging.getLogger(__name__)


class Agent(BaseModel):
    id: UUID4 = Field(default_factory=uuid.uuid4)
    alive: bool = True
    name: str
    parents: Optional[Tuple[UUID4, UUID4]] = None
    sexualPartners: List[UUID4] = Field(default_factory=list)
    index: int = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        logger.debug("Agent '%s' created with id %s.", self.name, self.id)

    """
    def __del__(self) -> None:
        logger.info("Agent '%s' with id %s has been deleted.", self.name, self.id)
    """

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Agent):
            return NotImplemented
        return self.id == other.id

    def __str__(self) -> str:
        parents_str = str(self.parents) if self.parents else "no parents"
        return f"{self.name} (id: {self.id}) with {parents_str}"

    def __repr__(self) -> str:
        return f"<BaseAgent name={self.name!r}, id={self.id!r}>"

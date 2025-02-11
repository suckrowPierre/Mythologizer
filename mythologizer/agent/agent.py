from __future__ import annotations
import uuid
import logging
from typing import List, Optional, Any, Tuple, Set
from pydantic import BaseModel, Field, UUID4

from mythologizer.memory import Memory

logger = logging.getLogger(__name__)


class Agent(BaseModel):
    id: UUID4 = Field(default_factory=uuid.uuid4)
    name: str
    parents: Optional[Tuple[UUID4, UUID4]] = None
    index: int = None
    culture_ids: Set[UUID4] = Field(default_factory=set)
    memory: Memory = Memory(size=10)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        logger.debug("Agent '%s' created with id %s.", self.name, self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Agent):
            return NotImplemented
        return self.id == other.id

    """
    def __str__(self) -> str:
        parents_str = str(self.parents) if self.parents else "no parents"
        return f"{self.name} (id: {self.id}) with {parents_str}"
    """

    def __repr__(self) -> str:
        return f"<BaseAgent name={self.name!r}, id={self.id!r}>"

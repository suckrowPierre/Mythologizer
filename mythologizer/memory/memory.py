import logging
from typing import List

import pydantic
from pydantic import BaseModel, Field, ConfigDict

from mythologizer.myths import Myth

logger = logging.getLogger(__name__)


class Memory(BaseModel):
    """
    A container class for storing and managing a list of Myth objects
    with a fixed maximum size. When a new Myth is added, it becomes
    the 'freshest' item at the front of the list.
    """

    size: int = Field(..., gt=0, description="Maximum number of myths to store.")
    myths: List[Myth] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def add_myth(self, myth: Myth) -> None:
        """
        Insert a Myth at the front of the list. If memory is at capacity,
        remove the myth at the end of the list.
        """
        logger.debug(f"Adding myth: {myth}")
        if len(self.myths) >= self.size:
            removed_myth = self.myths.pop(-1)
            logger.debug(f"Memory full. Removed oldest myth: {removed_myth}")

        # Insert the new myth at index 0, making it the freshest.
        self.myths.insert(0, myth)
        logger.debug(f"Myth added to the front. Current number of myths: {len(self.myths)}")

    def reorder_myths(self) -> None:
        """
        Reorder the list of myths by descending myth.retention so that
        the item with the highest retention is at the front.
        Python's sort is stable, so items with equal retention
        will maintain their relative order.
        """
        logger.debug("Reordering myths based on retention (descending).")
        self.myths.sort(key=lambda m: m.retention, reverse=True)
        logger.debug("Myths reordered. The first item now has the highest retention.")

    def change_memory_size(self, new_size: int) -> None:
        """
        Change the maximum allowed size for the myths list.
        If the new size is smaller than the current number of myths,
        truncate from the end.
        """
        logger.debug(f"Changing memory size from {self.size} to {new_size}.")
        self.size = new_size
        if len(self.myths) > new_size:
            self.myths = self.myths[:new_size]
            logger.debug("Truncated myths list to fit the new size.")
        logger.debug("Memory size changed successfully.")

    def __len__(self) -> int:
        return len(self.myths)

    def __getitem__(self, index: int) -> Myth:
        return self.myths[index]

    def __iter__(self):
        return iter(self.myths)

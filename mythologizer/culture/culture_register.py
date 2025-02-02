import logging
from typing import List, Union, Iterator, TypeVar, Generic
from uuid import UUID
from pydantic import BaseModel, Field, UUID4, StrictInt

from .culture import Culture
from mythologizer.registry import Registry, KeyConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CultureRegistry(Registry[Culture]):
    def __init__(self, records: List[Culture] = []):
        key_configs = [
            KeyConfig(prop_name="names", attr_name="name", expected_type=str),
            KeyConfig(prop_name="ids", attr_name="id", expected_type=uuid.UUID),
        ]

        super().__init__(key_configs=key_configs, records=records)
        logger.debug("CultureRegistry initialized.")

    def affected_by_current_events(self, events):
        # TODO: Your existing or future logic goes here
        pass

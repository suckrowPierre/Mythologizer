import logging
from typing import List, Union, Iterator, TypeVar, Generic

from pydantic import BaseModel, Field, validator

from .agent_attribute import AgentAttribute

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _to_list(item_or_list: Union[T, List[T]]) -> List[T]:
    """
    Utility to ensure the input is a list.
    """
    return item_or_list if isinstance(item_or_list, list) else [item_or_list]


class AgentAttributeRegister(BaseModel):
    agent_attributes: List[AgentAttribute] = Field(default_factory=list)

    @validator('agent_attributes')
    def check_unique_names(cls, v):
        names = [attr.name for attr in v]
        duplicates = set(name for name in names if names.count(name) > 1)
        if duplicates:
            raise ValueError(f"Duplicate agent attribute names found: {duplicates}")
        return v

    def __init__(self, **data):
        super().__init__(**data)
        logger.debug(
            f"AgentAttributeRegister initialized with {len(self.agent_attributes)} agent_attributes."
        )

    @property
    def names(self) -> List[str]:
        """Retrieve the names of all agent attributes."""
        return [attr.name for attr in self.agent_attributes]

    @property
    def values(self) -> List[T]:
        """Retrieve the values of all agent attributes."""
        return [attr.value for attr in self.agent_attributes]

    @property
    def descriptions(self) -> List[str]:
        """Retrieve the descriptions of all agent attributes."""
        return [attr.description for attr in self.agent_attributes]

    def __str__(self) -> str:
        return f"AgentAttributeRegister(agent_attributes={str(self.agent_attributes)})"

    def __repr__(self) -> str:
        return self.__str__()

    def add_agent_attributes(
        self,
        agent_attribute_or_attributes: Union[AgentAttribute, List[AgentAttribute]],
    ) -> None:
        """
        Add one or multiple AgentAttribute instances to the register.

        Raises:
            ValueError: If any attribute name already exists.
        """
        new_attrs = _to_list(agent_attribute_or_attributes)
        existing_names = set(self.names)
        new_names = {attr.name for attr in new_attrs}

        duplicates = existing_names.intersection(new_names)
        if duplicates:
            raise ValueError(f"Agent attributes with names {duplicates} already exist.")

        self.agent_attributes.extend(new_attrs)
        logger.info(f"Added agent attributes: {[attr.name for attr in new_attrs]}")

    def remove_agent_attribute_by_index(
        self, index_or_indices: Union[int, List[int]]
    ) -> None:
        """
        Remove one or multiple AgentAttributes by their index.

        Raises:
            IndexError: If any index is out of bounds.
        """
        indices = sorted(_to_list(index_or_indices), reverse=True)
        for index in indices:
            if not 0 <= index < len(self.agent_attributes):
                logger.error(f"Index {index} is out of bounds.")
                raise IndexError(f"Index {index} is out of bounds.")
            removed_attr = self.agent_attributes.pop(index)
            logger.info(f"Removed AgentAttribute '{removed_attr.name}' at index {index}.")

    def get_agent_attribute_by_name(
        self, name_or_names: Union[str, List[str]]
    ) -> Union[AgentAttribute, List[AgentAttribute]]:
        """
        Retrieve one or multiple AgentAttributes by name.

        Raises:
            ValueError: If any name is not found.
        """
        names = _to_list(name_or_names)
        if len(names) == 1:
            return self._get_single_attribute(name=names[0])
        return [self._get_single_attribute(name) for name in names]

    def _get_single_attribute(self, name: str) -> AgentAttribute:
        matches = [attr for attr in self.agent_attributes if attr.name == name]
        if not matches:
            logger.error(f"AgentAttribute with name '{name}' not found.")
            raise ValueError(f"AgentAttribute with name '{name}' not found.")
        return matches[0]

    def __len__(self) -> int:
        return len(self.agent_attributes)

    def __iter__(self) -> Iterator[AgentAttribute]:
        return iter(self.agent_attributes)

    def __getitem__(
        self, key_or_keys: Union[int, str, List[Union[int, str]]]
    ) -> Union[AgentAttribute, List[AgentAttribute]]:
        keys = _to_list(key_or_keys)
        if len(keys) == 1:
            return self._get_item(keys[0])
        return [self._get_item(k) for k in keys]

    def _get_item(self, key: Union[int, str]) -> AgentAttribute:
        index = self._resolve_key(key)
        return self.agent_attributes[index]

    def __setitem__(
        self,
        key_or_keys: Union[int, str, List[Union[int, str]]],
        value_or_values: Union[AgentAttribute, List[AgentAttribute]],
    ) -> None:
        keys = _to_list(key_or_keys)
        values = _to_list(value_or_values)

        if len(keys) != len(values):
            logger.error(
                f"Number of keys ({len(keys)}) does not match number of values ({len(values)})."
            )
            raise ValueError(
                f"Number of keys ({len(keys)}) does not match number of values ({len(values)})."
            )

        for key, value in zip(keys, values):
            index = self._resolve_key(key)
            old_attr = self.agent_attributes[index]
            self.agent_attributes[index] = value
            logger.info(
                f"Set AgentAttribute at '{key}' from '{old_attr.name}' to '{value.name}'."
            )

    def __delitem__(
        self, key_or_keys: Union[int, str, List[Union[int, str]]]
    ) -> None:
        keys = _to_list(key_or_keys)
        int_keys = sorted(
            [k for k in keys if isinstance(k, int)], reverse=True
        )
        str_keys = [k for k in keys if isinstance(k, str)]

        # Remove integer keys first to prevent shifting
        for key in int_keys:
            if not 0 <= key < len(self.agent_attributes):
                logger.error(f"Index {key} is out of bounds.")
                raise IndexError(f"Index {key} is out of bounds.")
            removed_attr = self.agent_attributes.pop(key)
            logger.info(f"Deleted AgentAttribute '{removed_attr.name}' at index {key}.")

        # Remove string keys
        for key in str_keys:
            attr = self._get_single_attribute(key)
            self.agent_attributes.remove(attr)
            logger.info(f"Deleted AgentAttribute '{attr.name}' by name '{key}'.")

    def _resolve_key(self, key: Union[int, str]) -> int:
        if isinstance(key, int):
            if 0 <= key < len(self.agent_attributes):
                return key
            logger.error(f"Index {key} is out of bounds.")
            raise IndexError(f"Index {key} is out of bounds.")
        elif isinstance(key, str):
            attr = self._get_single_attribute(key)
            return self.agent_attributes.index(attr)
        else:
            logger.error("Key must be an integer or string.")
            raise TypeError("Key must be an integer or string.")

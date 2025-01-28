import logging
from typing import List, Union, Iterator, TypeVar, Generic
from uuid import UUID

from pydantic import BaseModel, Field, UUID4, StrictInt

from mythologizer.culture import Culture

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _to_list(item_or_list: Union[T, List[T]]) -> List[T]:
    """
    Utility to unify single or multiple inputs into a list of T.
    """
    return item_or_list if isinstance(item_or_list, list) else [item_or_list]


class CultureRegister(BaseModel):
    cultures: List[Culture] = Field(default_factory=list)

    def __init__(self, **data):
        """
        Initialize a CultureRegister instance and log its creation.

        Args:
            **data: Arbitrary keyword arguments corresponding to the model fields.
        """
        super().__init__(**data)
        logger.debug(f"CultureRegister initialized with {len(self.cultures)} cultures.")

    @property
    def names(self) -> List[str]:
        """
        Retrieve the names of all cultures in the register.
        """
        return [culture.name for culture in self.cultures]

    @property
    def ids(self) -> List[UUID4]:
        """
        Retrieve the IDs of all cultures in the register.
        """
        return [culture.id for culture in self.cultures]

    @property
    def descriptions(self) -> List[str]:
        """
        Retrieve the descriptions of all cultures in the register.
        """
        return [culture.description for culture in self.cultures]

    def __str__(self) -> str:
        """
        String representation of the CultureRegister.
        """
        return f"Culture register with cultures: {', '.join(self.names)}"

    def __repr__(self) -> str:
        """
        Official string representation of the CultureRegister.
        """
        cultures_repr = ", ".join(repr(culture) for culture in self.cultures)
        return f"CultureRegister(cultures=[{cultures_repr}])"

    #
    # ADD CULTURE
    #
    def add_culture(self, culture_or_cultures: Union[Culture, List[Culture]]) -> None:
        """
        Add one or multiple Culture instances to the register.

        Args:
            culture_or_cultures: A single Culture or a list of Cultures.
        """
        cultures_list = _to_list(culture_or_cultures)
        self.cultures.extend(cultures_list)
        for c in cultures_list:
            logger.info(f"Added culture '{c.name}' with ID {c.id} to register.")

    #
    # REMOVE CULTURE
    #
    def remove_culture_by_id(self, id_or_ids: Union[UUID4, List[UUID4]]) -> None:
        """
        Remove one or multiple cultures from the register by their UUID.

        Args:
            id_or_ids: A single UUID or a list of UUIDs to remove.

        Raises:
            IndexError / ValueError: If any UUID is not found.
        """
        ids_list = _to_list(id_or_ids)
        for cid in ids_list:
            logger.debug(f"Attempting to remove culture with ID {cid}.")
            index = self._resolve_key(cid)
            removed_culture = self.cultures.pop(index)
            logger.info(f"Removed culture '{removed_culture.name}' with ID {cid} from register.")

    def remove_culture_by_index(self, index_or_indices: Union[StrictInt, List[StrictInt]]) -> None:
        """
        Remove one or multiple cultures from the register by their index.

        Args:
            index_or_indices: A single index or a list of indexes to remove.

        Raises:
            IndexError: If any index is out of bounds.
        """
        idxs = _to_list(index_or_indices)
        # Removing from higher indices first avoids shifting problems
        for i in sorted(idxs, reverse=True):
            logger.debug(f"Attempting to remove culture at index {i}.")
            removed_culture = self.cultures.pop(i)
            logger.info(f"Removed culture '{removed_culture.name}' at index {i} from register.")

    #
    # GET CULTURE / GET INDEX
    #
    def get_culture_by_id(self, id_or_ids: Union[UUID4, List[UUID4]]) -> Union[Culture, List[Culture]]:
        """
        Retrieve one or multiple cultures by UUID(s).

        Args:
            id_or_ids: A single UUID or a list of UUIDs.

        Returns:
            A single Culture or a list of Cultures.

        Raises:
            ValueError: If any UUID is not found in the register.
        """
        ids_list = _to_list(id_or_ids)
        # If only one item, return single Culture.
        if len(ids_list) == 1:
            return self._get_culture_by_id_single(ids_list[0])
        # Otherwise return list of results.
        return [self._get_culture_by_id_single(cid) for cid in ids_list]

    def _get_culture_by_id_single(self, cid: UUID4) -> Culture:
        """
        Helper to get a single Culture by ID or raise ValueError if not found.
        """
        for culture in self.cultures:
            if culture.id == cid:
                return culture
        message = f"Culture with ID {cid} is not present in the register."
        logger.error(message)
        raise ValueError(message)

    def get_index_by_id(self, id_or_ids: Union[UUID4, List[UUID4]]) -> Union[int, List[int]]:
        """
        Get the index (or indices) of one or multiple cultures by UUID(s).

        Args:
            id_or_ids: A single UUID or a list of UUIDs.

        Returns:
            Single index or a list of indices.

        Raises:
            ValueError: If any UUID is not found in the register.
        """
        ids_list = _to_list(id_or_ids)
        if len(ids_list) == 1:
            return self._get_index_by_id_single(ids_list[0])
        return [self._get_index_by_id_single(cid) for cid in ids_list]

    def _get_index_by_id_single(self, cid: UUID4) -> int:
        """
        Helper to get a single index by ID or raise ValueError if not found.
        """
        for index, culture in enumerate(self.cultures):
            if culture.id == cid:
                return index
        message = f"Culture with ID {cid} is not present in the register."
        logger.error(message)
        raise ValueError(message)

    def get_culture_by_name(self, name_or_names: Union[str, List[str]]) -> Union[Culture, List[Culture]]:
        """
        Retrieve one or multiple cultures by name.

        - Raises an error if not found or if duplicates exist (for single name).

        Args:
            name_or_names: A single culture name or a list of names.

        Returns:
            A single Culture or a list of Cultures.

        Raises:
            ValueError: If a name is not present or if multiple matches are found (for single name).
        """
        names_list = _to_list(name_or_names)
        if len(names_list) == 1:
            return self._get_culture_by_name_single(names_list[0])
        return [self._get_culture_by_name_single(nm) for nm in names_list]

    def _get_culture_by_name_single(self, name: str) -> Culture:
        """
        Helper to get a single Culture by name or raise ValueError if not found or if multiple found.
        """
        matches = [culture for culture in self.cultures if culture.name == name]
        if not matches:
            message = f"Culture with name '{name}' is not present in the register."
            logger.error(message)
            raise ValueError(message)
        if len(matches) > 1:
            message = f"Multiple cultures with name '{name}' are present in the register."
            logger.error(message)
            raise ValueError(message)
        return matches[0]

    #
    # DUNDER METHODS: LEN, ITER
    #
    def __len__(self) -> int:
        """
        Return the number of cultures in the register.
        """
        return len(self.cultures)

    def __iter__(self) -> Iterator[Culture]:
        """
        Return an iterator over all cultures.
        """
        return iter(self.cultures)

    #
    # DUNDER METHODS: GETITEM, SETITEM, DELITEM
    #
    def __getitem__(
        self,
        key_or_keys: Union[int, UUID, str, List[Union[int, UUID, str]]]
    ) -> Union[Culture, List[Culture]]:
        """
        Get one or multiple cultures by index, UUID, or name.

        Args:
            key_or_keys: A single key (int / UUID / str) or a list of keys.

        Returns:
            A single Culture or a list of Cultures.

        Raises:
            KeyError, IndexError, or TypeError: If a key is invalid or out of bounds.
        """
        keys = _to_list(key_or_keys)
        if len(keys) == 1:
            return self._get_single_item(keys[0])
        return [self._get_single_item(k) for k in keys]

    def _get_single_item(self, key: Union[int, UUID, str]) -> Culture:
        """
        Helper to retrieve a single Culture by an int, UUID, or str key.
        Raises KeyError or IndexError if not found.
        """
        index = self._resolve_key(key)
        return self.cultures[index]

    def __setitem__(
        self,
        key_or_keys: Union[int, UUID, str, List[Union[int, UUID, str]]],
        value_or_values: Union[Culture, List[Culture]]
    ) -> None:
        """
        Set one or multiple cultures in the register by index, UUID, or name.

        Args:
            key_or_keys: A single key or list of keys (int / UUID / str).
            value_or_values: A single Culture or list of Cultures.

        Raises:
            KeyError, IndexError, or TypeError: If the key is invalid or the lengths do not match.
        """
        keys = _to_list(key_or_keys)
        values = _to_list(value_or_values)

        if len(keys) != len(values):
            msg = (
                f"Number of keys ({len(keys)}) does not match "
                f"number of values ({len(values)})"
            )
            logger.error(msg)
            raise ValueError(msg)

        for k, v in zip(keys, values):
            idx = self._resolve_key(k)
            old_culture = self.cultures[idx]
            self.cultures[idx] = v
            key_type = type(k).__name__
            logger.info(
                f"Set culture at {key_type} '{k}' from '{old_culture.name}' to '{v.name}'."
            )

    def __delitem__(
        self,
        key_or_keys: Union[int, UUID, str, List[Union[int, UUID, str]]]
    ) -> None:
        """
        Delete one or multiple cultures by index, UUID, or name.

        Args:
            key_or_keys: Single key or a list of keys (int / UUID / str).

        Raises:
            KeyError, IndexError, or TypeError: If any key is invalid or out of bounds.
        """
        keys = _to_list(key_or_keys)
        # Delete from largest index first if int-keys to avoid shifting issues.
        # For string/UUID we can just do them in a loop; the order doesn't matter.
        # We'll handle int keys separately and do them in descending order, then
        # handle non-int keys as they come.
        int_keys = [(k, idx) for idx, k in enumerate(keys) if isinstance(k, int)]
        non_int_keys = [(k, idx) for idx, k in enumerate(keys) if not isinstance(k, int)]

        # Sort int keys by descending value to avoid reindexing issues
        int_keys_sorted = sorted(int_keys, key=lambda x: x[0], reverse=True)

        # We'll reconstruct the keys in the order that won't break indexing
        reordered_keys = int_keys_sorted + non_int_keys

        for key, _ in reordered_keys:
            index = self._resolve_key(key)
            removed_culture = self.cultures.pop(index)
            key_type = type(key).__name__
            logger.info(f"Deleted culture at {key_type} '{key}': '{removed_culture.name}'.")

    def _resolve_key(self, key: Union[int, UUID, str]) -> int:
        """
        Resolve a key (int, UUID, or string) to a culture index.

        Raises:
            IndexError: if integer key is out of range
            KeyError: if UUID or string key is not found
            TypeError: if key type is unsupported
        """
        if isinstance(key, int):
            if 0 <= key < len(self.cultures):
                return key
            message = f"Index {key} is out of bounds."
            logger.error(message)
            raise IndexError(message)
        elif isinstance(key, UUID):
            return self._get_index_by_id_single(key)
        elif isinstance(key, str):
            # Reuse our name-based retrieval
            culture = self._get_culture_by_name_single(key)
            return self.cultures.index(culture)
        else:
            message = "Key must be an integer, UUID, or string."
            logger.error(message)
            raise TypeError(message)

    def affected_by_current_events(self, events):
        # TODO: Your existing or future logic goes here
        pass

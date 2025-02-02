import logging
import uuid
from typing import Any, List, Optional, Type, TypeVar, Generic, Union
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)
T = TypeVar("T")


def ensure_list(item_or_list: Union[T, List[T]]) -> List[T]:
    """Return a list, wrapping the argument if necessary."""
    return item_or_list if isinstance(item_or_list, list) else [item_or_list]


class KeyConfig(BaseModel):
    prop_name: str         # Name for dynamic property access (e.g. "names")
    attr_name: str         # Attribute on each record (e.g. "name")
    expected_type: Type[Any]  # Expected type (e.g. str, uuid.UUID)

    @validator("prop_name")
    def validate_prop_name(cls, v: str) -> str:
        if not v.isidentifier() or v.isdigit():
            raise ValueError(f"Invalid prop_name '{v}'; must be a valid non-numeric identifier.")
        return v


class Registry(BaseModel, Generic[T]):
    key_configs: List[KeyConfig]
    records: List[T] = Field(default_factory=list)

    @validator("key_configs")
    def _validate_key_configs(cls, configs: List[KeyConfig]) -> List[KeyConfig]:
        seen = {}
        for cfg in configs:
            if cfg.expected_type != uuid.UUID:
                if cfg.expected_type in seen:
                    raise ValueError(
                        f"Duplicate expected type {cfg.expected_type} used for '{seen[cfg.expected_type]}' and '{cfg.prop_name}'."
                    )
                seen[cfg.expected_type] = cfg.prop_name
        return configs

    def __init__(self, **data: Any):
        super().__init__(**data)
        rec_type = self.records[0].__class__.__name__ if self.records else "Unknown"
        logger.debug(f"Registry initialized for {rec_type}.")

    def __getattr__(self, name: str) -> Any:
        """Dynamically expose each key config as a property."""
        for cfg in self.key_configs:
            if cfg.prop_name == name:
                values = [getattr(rec, cfg.attr_name) for rec in self.records]
                logger.debug(f"Accessed dynamic property '{name}': {values}")
                return values
        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")

    def __str__(self) -> str:
        rec_type = self.records[0].__class__.__name__ if self.records else "Unknown"
        keys = ", ".join(cfg.prop_name for cfg in self.key_configs)
        return f"Registry<{rec_type}>: {len(self.records)} records, keys: [{keys}]"

    def _check_uniqueness(self, record: T, exclude_index: Optional[int] = None) -> None:
        """
        Assert that for each key (except UUID) the record's attribute value is unique.
        When updating, exclude the record at `exclude_index` from the check.
        """
        for cfg in self.key_configs:
            if cfg.expected_type != uuid.UUID:
                new_val = getattr(record, cfg.attr_name)
                for i, existing in enumerate(self.records):
                    if exclude_index is not None and i == exclude_index:
                        continue
                    if getattr(existing, cfg.attr_name) == new_val:
                        logger.debug(f"Duplicate detected for key '{cfg.prop_name}' with value {new_val}.")
                        raise ValueError(f"Duplicate value for key '{cfg.prop_name}': {new_val}")

    def append(self, record: Union[T, List[T]]) -> None:
        """
        Append one or more records to the registry.
        Each record is checked for uniqueness (for non-UUID keys).
        """
        for rec in ensure_list(record):
            self._check_uniqueness(rec)
            self.records.append(rec)
            logger.debug(f"Appended record: {rec}. Total records: {len(self.records)}.")

    def resolve_index_by_key(self, key_value: Any) -> Optional[int]:
        """
        Given a key value, return the index of the matching record.
        If no match is found, return None. Raise an error if multiple records match.
        """
        for cfg in self.key_configs:
            if isinstance(key_value, cfg.expected_type):
                indices = [i for i, rec in enumerate(self.records)
                           if getattr(rec, cfg.attr_name) == key_value]
                if not indices:
                    logger.debug(f"No record found with {cfg.prop_name} == {key_value}.")
                    return None
                if len(indices) > 1:
                    raise ValueError(f"Ambiguous key value {key_value} for key '{cfg.prop_name}'; multiple records found.")
                logger.debug(f"Resolved key {key_value} to index {indices[0]} using key '{cfg.prop_name}'.")
                return indices[0]
        logger.debug(f"No key config found matching type {type(key_value)} for value {key_value}.")
        return None

    def _resolve_index(self, key: Union[int, Any]) -> int:
        """Return the numeric index corresponding to an int or key value."""
        if isinstance(key, int):
            if key < 0 or key >= len(self.records):
                raise IndexError(f"Index {key} out of range.")
            return key
        idx = self.resolve_index_by_key(key)
        if idx is None:
            raise KeyError(f"Key {key} not found.")
        return idx

    def _resolve_indices(self, keys: Union[Any, List[Any]]) -> List[int]:
        """Return a list of indices corresponding to a single key or list of keys."""
        return [self._resolve_index(k) for k in ensure_list(keys)]

    def __getitem__(self, key: Union[int, Any, List[Any]]) -> Union[T, List[T]]:
        indices = self._resolve_indices(key)
        return self.records[indices[0]] if len(indices) == 1 else [self.records[i] for i in indices]

    def __setitem__(self, key: Union[int, Any, List[Any]], value: Union[T, List[T]]) -> None:
        indices = self._resolve_indices(key)
        values = ensure_list(value)
        if len(indices) != len(values):
            raise ValueError("Number of keys and values must match.")
        for i, val in zip(indices, values):
            self._check_uniqueness(val, exclude_index=i)
            self.records[i] = val
            logger.debug(f"Updated record at index {i} to {val}.")

    def __delitem__(self, key: Union[int, Any, List[Any]]) -> None:
        indices = sorted(self._resolve_indices(key), reverse=True)
        for i in indices:
            rec = self.records.pop(i)
            logger.debug(f"Deleted record at index {i}: {rec}. Total records: {len(self.records)}.")

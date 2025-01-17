import uuid
from abc import ABC, abstractmethod
from typing import List, Optional


class BaseAgent:

    def __init__(self, name: str, parent_a: Optional["BaseAgent"] = None, parent_b: Optional["BaseAgent"] = None):
        self.id = uuid.uuid4()
        self.name = name
        self.memory = List[str]  # For now strings TODO: make class for memories
        self.attributes = List[str]  # For now strings TODO: make class for attributes
        self.parent_a = parent_a
        self.parent_b = parent_b

    #destructor
    def __del__(self):
        print(f"{self.name} with id {self.id} has died.")

    def kill(self):
        del self

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return self.id != other.id

    def __str__(self):
        if self.parent_a is None and self.parent_b is None:
            return f"{self.name} with id {self.id} created by god"
        return f"{self.name} with id {self.id} and parents {self.parent_a} and {self.parent_b}"

if __name__ == "__main__":
    agent = BaseAgent("Bob")
    print(agent)
    del agent
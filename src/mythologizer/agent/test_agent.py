import uuid
from typing import List

from mythologizer.agent_attribute import AgentAttribute


class TestAgent:
    def __init__(self, name: str, agent_attributes: List[AgentAttribute]):
        self.id = uuid.uuid4()
        self.name = name
        self.attributes = agent_attributes

    def __del__(self):
        print(f"{self.name} with id {self.id} has died.")

    def kill(self):
        del self

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return self.id != other.id

    def __str__(self):
        return f"{self.name} with id {self.id} created by god"

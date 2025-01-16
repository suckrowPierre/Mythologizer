__version__ = "0.0.1"
__author__ = "Lilli Kurth, Cele Meunier, Eman Safavi, Pierre-Louis Suckrow"

from .agent_attribute import (
    AgentAttribute,
    MutableAgentAttribute,
    ConstantAgentAttribute,
    DynamicAgentAttribute,
    IteratingAgentAttribute,
)

from .agent import (
    BaseAgent,
    TestAgent,
)

__all__ = [
    "AgentAttribute",
    "MutableAgentAttribute",
    "ConstantAgentAttribute",
    "DynamicAgentAttribute",
    "IteratingAgentAttribute",
    "BaseAgent",
    "TestAgent"
]


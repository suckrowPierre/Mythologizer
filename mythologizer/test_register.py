from mythologizer.culture import Culture, CultureRegistry, AttributeDistribution, AttributesDistributions
from mythologizer.random_number_generator import RandomNumberGenerator as RNG
from mythologizer.llm import gtp4o_culture_agent_attribute_distribution_map
from mythologizer.agent_attribute import AgentAttribute, AgentAttributeRegistry
from mythologizer.agent import Agent
from mythologizer.registry import Registry, KeyConfig
import logging
from openai import OpenAI
from dotenv import load_dotenv
import uuid
import os

if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    agent_keys = [
        KeyConfig(prop_name="names", attr_name="name", expected_type=str),
        KeyConfig(prop_name="ids", attr_name="id", expected_type=uuid.UUID),
    ]

    # Create a register for Agent objects.
    agent_registry = Registry[Agent](key_configs=agent_keys)

    bob = Agent(name="bob")
    larry = Agent(name="larry")

    agent_registry.append([bob,larry])

    # Access dynamic properties. These are generated on the fly.
    print("Agent names:", agent_registry.names)  # Outputs: ['Alice', 'Bob']
    print("Agent ids:", agent_registry.ids)      # Outputs: [UUID(...), UUID(...)]
    print("Agent parents:", agent_registry.parents)

    # Use resolve_key_to_index() to find an agent's index by its name.
    index_bob = agent_registry.resolve_index_by_key("bob")
    print("Index for agent named 'Alice':", index_bob)

    # Also resolve using an agent's UUID.
    index_agent2 = agent_registry.resolve_index_by_key(larry.id)
    print(f"Index for agent with id {larry.id}: {index_agent2}")

    # Print a summary of the register.
    print(agent_registry["larry"])




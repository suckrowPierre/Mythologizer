from mythologizer.culture import Culture, AttributeDistribution, CultureRegistry
from mythologizer.random_number_generator import RandomNumberGenerator as RNG
from mythologizer.llm import gtp4o_culture_agent_attribute_distribution_map
from mythologizer.agent_attribute import AgentAttribute
from mythologizer.agent import Agent
from mythologizer.registry import Registry, KeyConfig
from mythologizer.population import Population
from mythologizer.agent_attribute_matrix import AgentAttributeMatrix
from mythologizer.population_handler import AgentLifecycleManager
import logging
from openai import OpenAI
from dotenv import load_dotenv
import uuid
import os
import numpy as np

if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')



    tech_bros = Culture(name="Tech bros", description="naughty boys")
    haexxen = Culture(name="Häxxen", description="naughty witches")
    weed_heads = Culture(name="Weed heads", description="420")
    cultures = [tech_bros, haexxen, weed_heads]

    anna = Agent(name="anna", culture_ids={haexxen.id})
    bob = Agent(name="bob", culture_ids={tech_bros.id, weed_heads.id})

    agents = [anna, bob]

    culture_register = CultureRegistry(records=cultures)
    print(culture_register[bob.culture_ids])

from mythologizer.culture import Culture, AttributeDistribution
from mythologizer.random_number_generator import RandomNumberGenerator as RNG
from mythologizer.llm import gtp4o_culture_agent_attribute_distribution_map
from mythologizer.agent_attribute import AgentAttribute
from mythologizer.agent import Agent
from mythologizer.registry import Registry, KeyConfig
from mythologizer.population import Population
from mythologizer.agent_attribute_matrix import AgentAttributeMatrix
from mythologizer.population_handler import PopulationHandler
import logging
from openai import OpenAI
from dotenv import load_dotenv
import uuid
import os

if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    population = Population()
    print(population.agents)
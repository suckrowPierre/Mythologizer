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
import numpy as np

if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    anna = Agent(name="anna")
    bob = Agent(name="bob")
    carsten = Agent(name="carsten")
    daniel = Agent(name="daniel")
    emil = Agent(name="emil")
    fabian = Agent(name="fabian")
    gustav = Agent(name="gustav")
    heinrich = Agent(name="heinrich")
    ingrid = Agent(name="ingrid")
    juergen = Agent(name="juergen")
    konstantin = Agent(name="konstantin")
    larry = Agent(name="larry")
    mona = Agent(name="mona")
    niels = Agent(name="niels")
    otto = Agent(name="otto")

    agents = [anna, bob, carsten, daniel, emil, fabian, gustav, heinrich, ingrid, juergen, konstantin, larry, mona,
              niels, otto]


    speed = AgentAttribute(name='Speed', description='Speed', d_type=float, min=0.)
    health = AgentAttribute(name='Health', description='Health', d_type=float)
    confidence = AgentAttribute(name='Confidence', description='Confidence', d_type=float)

    attributes = [speed, health, confidence]

    population_handler = PopulationHandler(agents=agents, agent_attributes=attributes)
    print(population_handler.agent_attribute_matrix.matrix)
    print(f"index of agent larry in matrix {population_handler.get_index_from_agent_uuid(larry.id)}")
    population_handler.kill_agent(konstantin)
    print(f"index of agent larry in matrix {population_handler.get_index_from_agent_uuid(larry.id)}")
    print(f"alive index list: {population_handler.alive_agents_uuids} with larry having the id {larry.id}")
    print(f"dead agents: {population_handler.population.dead_agents}")

    pierre = Agent(name="pierre")
    pierre_values = np.array([1.0, 1.0, 1.0])
    population_handler.add_agent(agent=pierre, values=pierre_values)
    print(population_handler.agent_attribute_matrix.matrix)
    population_handler.agent_attribute_matrix.matrix["Speed"] += 1.0
    print(population_handler.agent_attribute_matrix.matrix[-1])
    print(population_handler.agent_attribute_matrix.matrix)

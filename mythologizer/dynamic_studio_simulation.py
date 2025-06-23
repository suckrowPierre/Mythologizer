import json
import os
import logging

from dotenv import load_dotenv

from mythologizer.culture import Culture
from mythologizer.agent_attribute import AgentAttribute
from mythologizer.agent import Agent
from mythologizer.myths import Myth
from mythologizer.memory import Memory
from mythologizer.population_handler import AgentLifecycleManager
from mythologizer.population import Population
from mythologizer.myth_exchange import tell_myth
from mythologizer.llm import ollame_interaction_pair
from typing import Any, Dict, Optional, List, Tuple
import random
import numpy as np
from openai import OpenAI


def epoch_iterate(
        values: np.ndarray, min_val: Optional[Any] = None, max_val: Optional[Any] = None
) -> np.ndarray:
    """
    Epoch function that increments the value by 1.
    If min_val or max_val are provided, the result is clamped accordingly.
    """
    new_values = values + 1
    if min_val is not None or max_val is not None:
        lower = min_val if min_val is not None else -np.inf
        upper = max_val if max_val is not None else np.inf
        new_values = np.clip(new_values, lower, upper)
    return new_values


def epoch_random_fluctuation(
        values: np.ndarray, min_val: Optional[Any] = None, max_val: Optional[Any] = None
) -> np.ndarray:
    """
    Epoch function that applies random fluctuation.
    It samples from a normal distribution centered at the current value.
    By default, the result is clamped between 0 and 1, but if min_val and max_val are provided,
    they are used instead. The standard deviation is set to 10% of the provided range (or 0.1 by default).
    """
    if min_val is not None and max_val is not None:
        std = (max_val - min_val) * 0.1
        lower, upper = min_val, max_val
    else:
        std = 0.1
        lower, upper = 0, 1
    new_values = np.random.normal(loc=values, scale=std)
    return np.clip(new_values, lower, upper)


EPOCH_FUNCTIONS = {
    "epoch_iterate": epoch_iterate,
    "epoch_random_fluctuation": epoch_random_fluctuation
}

def load_data_from_json(json_path: str) -> Dict[str, Any]:
    """Utility function to load JSON data from a file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_cultures(culture_data: list[dict]) -> Dict[str, Culture]:
    """Given a list of culture dicts, return a mapping from ID/name -> Culture object."""
    culture_map = {}
    for cdict in culture_data:
        # Using ID from JSON if present, or a fallback
        culture_id = cdict["name"]
        culture_obj = Culture(
            name=cdict["name"],
            description=cdict["description"]
            # If you have other fields, pass them here
        )
        # Remember the ID -> Culture object
        culture_map[culture_id] = culture_obj
    return culture_map


def build_attributes(attribute_data: list[dict]) -> list[AgentAttribute]:
    """Given a list of attribute dicts, return a list of AgentAttribute objects
    in the same order as in the JSON."""
    attributes = []
    for adict in attribute_data:
        func_name = adict.get("epoch_change_function")
        epoch_func = EPOCH_FUNCTIONS.get(func_name) if func_name else None

        # The `type` field in JSON might be "int" or "float"
        d_type = int if adict["type"] == "int" else float

        attr_obj = AgentAttribute(
            name=adict["name"],
            description=adict["description"],
            d_type=d_type,
            min=adict.get("min"),
            max=adict.get("max"),
            epoch_change_function=epoch_func
        )
        attributes.append(attr_obj)
    return attributes


def build_agents(
        agent_data: list[dict],
        culture_map: Dict[str, Culture],
        memory_size: int = 10
) -> list[Agent]:
    """Given a list of agent dicts and a culture map, build Agent objects."""
    agents = []
    for i, adict in enumerate(agent_data):
        # Resolve culture names to actual Culture ids
        culture_ids = {
            culture_map[cid].id for cid in adict["culture_ids"]
            if cid in culture_map
        }
        # Build Myth objects from JSON
        myths = []
        for m in adict.get("myths", []):
            myth_obj = Myth(
                current_myth=m["current_myth"],
                mythemes=set(m.get("mythemes", []))
            )
            myths.append(myth_obj)

        # If you store attribute_values in JSON, just parse them
        # but do NOT assign them to the Agent object directly.
        # Typically these go into your attribute matrix.
        # We'll still attach them as a separate field or keep them
        # so you can set them later in AgentLifecycleManager.
        attribute_values = adict.get("attribute_values", [])

        # Construct the agent’s memory
        agent_memory = Memory(size=memory_size, myths=myths)

        agent_obj = Agent(
            name=adict["name"],
            culture_ids=culture_ids,
            memory=agent_memory,

        )
        # We can store these values on the agent in a separate attribute, or
        # handle them in the next step. For example, do:
        agent_obj._raw_attribute_values = attribute_values  # custom for bridging

        agents.append(agent_obj)
    return agents


def main():
    logging.basicConfig(level=logging.DEBUG)

    data = load_data_from_json("studio_sim.json")

    culture_map = build_cultures(data["cultures"])
    cultures = list(culture_map.values())

    attributes = build_attributes(data["attributes"])

    agent_list = build_agents(
        data["agents"],
        culture_map=culture_map,
        memory_size=10
    )

    attribute_values = []
    for ag in agent_list:
        attribute_values.append(ag._raw_attribute_values)

    agent_lifecycle_manager = AgentLifecycleManager(
        agent_attributes=attributes,
        agents=agent_list,
        cultures=cultures,
        attribute_values=attribute_values
    )

    for agent in agent_list:
        del agent._raw_attribute_values

    def get_random_interaction_tuples(n_interactions: int, population: Population) -> List[Tuple]:
        def get_random_pair():
            x = random.choice(list(population.alive_agents.values()))
            y = random.choice(list(population.alive_agents.values()))
            while y == x:
                y = random.choice(list(population.alive_agents.values()))
            return x, y

        return [get_random_pair() for _ in range(n_interactions)]

    openai_client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

    # -------
    # define orignal mythes
    # TODO put them there from the website

    # define LLM client # TODO

    current_epoch = 0
    number_interactions = 10

    while current_epoch < 2:

        agent_lifecycle_manager.agent_attribute_matrix.apply_epoch_changing_functions()

        pairs = get_random_interaction_tuples(number_interactions, agent_lifecycle_manager.population)
        for pair in pairs:
            agent_a, agent_b = pair
            agent_a_values = agent_lifecycle_manager.agent_attribute_matrix.agent_attribute_register.create_values_dict(
                agent_lifecycle_manager.agent_attribute_matrix.matrix[agent_a.index])
            agent_b_values = agent_lifecycle_manager.agent_attribute_matrix.agent_attribute_register.create_values_dict(
                agent_lifecycle_manager.agent_attribute_matrix.matrix[agent_b.index])

            speaker, listener = ollame_interaction_pair(open_ai_client=openai_client, agent_A=agent_a,
                                                        agent_A_values=agent_a_values, agent_B=agent_b,
                                                        agent_B_values=agent_b_values,
                                                        culture_registry=agent_lifecycle_manager.culture_registry)
            if speaker == agent_a:
                speaker_values = agent_a_values
                listener_values = agent_b_values
            else:
                speaker_values = agent_b_values
                listener_values = agent_a_values

            if speaker is not None and listener is not None:
                logger.info(f"Interaction with {speaker} as a speaker and {listener} as a listener")
                tell_myth(
                    openai_client=openai_client,
                    culture_registry=agent_lifecycle_manager.culture_registry,
                    speaker_agent=speaker,
                    speaker_agent_values=speaker_values,
                    listener_agent=listener,
                    listener_agent_values=listener_values)

        current_epoch += 1
        logger.info(f"Current epoch: {current_epoch}")

if __name__ == "__main__":
    load_dotenv()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()

import json
import os
import logging
import random
import asyncio
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Import your simulation modules
from mythologizer.culture import Culture
from mythologizer.agent_attribute import AgentAttribute
from mythologizer.agent import Agent
from mythologizer.myths import Myth
from mythologizer.memory import Memory
from mythologizer.population_handler import AgentLifecycleManager
from mythologizer.population import Population
from mythologizer.myth_exchange import tell_myth
from mythologizer.llm import gtp4o_interaction_pair
from openai import OpenAI

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# --- Epoch Functions (as in your original code) ---

def epoch_iterate(
        values: np.ndarray, min_val: Optional[Any] = None, max_val: Optional[Any] = None
) -> np.ndarray:
    new_values = values + 1
    if min_val is not None or max_val is not None:
        lower = min_val if min_val is not None else -np.inf
        upper = max_val if max_val is not None else np.inf
        new_values = np.clip(new_values, lower, upper)
    return new_values


def epoch_random_fluctuation(
        values: np.ndarray, min_val: Optional[Any] = None, max_val: Optional[Any] = None
) -> np.ndarray:
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


# --- Utility Functions for building the simulation ---

def load_data_from_json(json_path: str) -> Dict[str, Any]:
    """Load JSON data from a file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_cultures(culture_data: List[dict]) -> Dict[str, Culture]:
    """Convert culture dictionaries into Culture objects."""
    culture_map = {}
    for cdict in culture_data:
        culture_id = cdict["name"]  # using the name as ID
        culture_obj = Culture(
            name=cdict["name"],
            description=cdict["description"]
        )
        culture_map[culture_id] = culture_obj
    return culture_map


def build_attributes(attribute_data: List[dict]) -> List[AgentAttribute]:
    """Convert attribute dictionaries into AgentAttribute objects."""
    attributes = []
    for adict in attribute_data:
        func_name = adict.get("epoch_change_function")
        epoch_func = EPOCH_FUNCTIONS.get(func_name) if func_name else None
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
        agent_data: List[dict],
        culture_map: Dict[str, Culture],
        memory_size: int = 10
) -> List[Agent]:
    """Convert agent dictionaries into Agent objects."""
    agents = []
    for i, adict in enumerate(agent_data):
        # Resolve culture names to actual Culture objects
        culture_ids = {
            culture_map[cid].id for cid in adict["culture_ids"]
            if cid in culture_map
        }
        # Build myth objects for the agent
        myths = []
        for m in adict.get("myths", []):
            myth_obj = Myth(
                current_myth=m["current_myth"],
                mythemes=set(m.get("mythemes", []))
            )
            myths.append(myth_obj)

        # Get raw attribute values (to be used later)
        attribute_values = adict.get("attribute_values", [])

        # Create the agent’s memory using the myths
        agent_memory = Memory(size=memory_size, myths=myths)

        # Create the Agent object
        agent_obj = Agent(
            name=adict["name"],
            culture_ids=culture_ids,
            memory=agent_memory,
        )
        # Temporarily store the attribute values to be passed to the lifecycle manager
        agent_obj._raw_attribute_values = attribute_values
        agents.append(agent_obj)
    return agents


# --- Global Variables for Simulation State ---

app = FastAPI()
agent_lifecycle_manager: Optional[AgentLifecycleManager] = None
current_epoch: int = 0


# --- Pydantic Model for API Response ---

class AgentResponse(BaseModel):
    name: str
    epoch: int
    attributes: Dict[str, float]
    mythemes: List[str]
    myth: Optional[str] = None


# --- Background Simulation Loop ---
# This function simulates one “epoch” every 10 seconds.
async def simulation_loop():
    global current_epoch, agent_lifecycle_manager
    while True:
        if agent_lifecycle_manager:
            # Update agent attributes (as in your simulation)
            agent_lifecycle_manager.agent_attribute_matrix.apply_epoch_changing_functions()
            current_epoch += 1
            logger.info(f"Simulation epoch: {current_epoch}")
        await asyncio.sleep(10)  # Adjust the sleep duration as needed


# --- Startup Event: Initialize the Simulation ---

@app.on_event("startup")
async def startup_event():
    load_dotenv()  # Load environment variables (e.g., OPENAI_KEY)
    global agent_lifecycle_manager, current_epoch
    # Load simulation data from JSON
    data = load_data_from_json("studio_sim.json")
    culture_map = build_cultures(data["cultures"])
    cultures = list(culture_map.values())
    attributes = build_attributes(data["attributes"])
    agent_list = build_agents(data["agents"], culture_map, memory_size=10)
    # Collect raw attribute values (this will be used to initialize the attribute matrix)
    attribute_values = []
    for ag in agent_list:
        attribute_values.append(ag._raw_attribute_values)
    # Initialize the lifecycle manager
    agent_lifecycle_manager = AgentLifecycleManager(
        agent_attributes=attributes,
        agents=agent_list,
        cultures=cultures,
        attribute_values=attribute_values
    )
    # Clean up the temporary attribute storage on each agent
    for agent in agent_list:
        del agent._raw_attribute_values
    current_epoch = 0
    # Start the background simulation loop
    asyncio.create_task(simulation_loop())


# --- API Endpoint to Retrieve an Agent by Name ---

@app.get("/agent/{agent_name}", response_model=AgentResponse)
async def get_agent(agent_name: str):
    global current_epoch, agent_lifecycle_manager
    if not agent_lifecycle_manager:
        raise HTTPException(status_code=500, detail="Simulation not initialized")
    # Look for the agent (using case-insensitive comparison)
    agent_found = agent_lifecycle_manager.population[agent_name]
    if not agent_found:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Retrieve the agent’s attribute values from the simulation matrix.
    # (This assumes that each agent has an `index` attribute; adjust if necessary.)
    try:
        index = agent_found.index
    except AttributeError:
        raise HTTPException(status_code=500, detail="Agent index not available")

    attr_values = agent_lifecycle_manager.agent_attribute_matrix.agent_attribute_register.create_values_dict(
        agent_lifecycle_manager.agent_attribute_matrix.matrix[index]
    )

    # Get myth data from the agent’s memory (using the first myth as an example)
    myths = agent_found.memory.myths
    if myths:
        first_myth = myths[0]
        myth = first_myth.current_myth
        mythemes = list(first_myth.mythemes)
    else:
        myth = None
        mythemes = []

    return AgentResponse(
        name=agent_found.name,
        epoch=current_epoch,
        attributes=attr_values,
        mythemes=mythemes,
        myth=myth
    )


# --- Main Block to Run the FastAPI Server ---

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

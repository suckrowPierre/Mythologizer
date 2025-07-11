from typing import Union, Literal, List, Dict
from pydantic import BaseModel, Field, conint, confloat, create_model, UUID4
from ollama import chat
import numpy as np

from mythologizer.agent import Agent
from mythologizer.culture import Culture, CultureRegistry



def ollame_interaction_pair(
        agent_A: Agent,
        agent_A_values: List[dict],
        agent_B: Agent,
        agent_B_values: List[dict],
        culture_registry: CultureRegistry
):

    def get_user_prompt():
        return f"""
You are an expert in statistical modeling and cultural analysis.

The following two agents just met. Based on their cultures aswell as their attribute values choose if the agents interact. If they interact they well exchange a story. For this choose a one of the agents as a speaker and one as a listener. Use the name to identify them. If no interaction is happing set both to an empty string.

Agent A id: {agent_A.id}
Agent A name: {agent_A.name}
Agent A cultures: {[f"{culture.name}:{culture.description}" for culture in culture_registry[agent_A.culture_ids]]}
Agent A values: {agent_A_values}
----
Agent B id: {agent_B.id}
Agent B name: {agent_B.name}
Agent B cultures: {[f"{culture.name}:{culture.description}" for culture in culture_registry[agent_B.culture_ids]]}
Agent B values: {agent_B_values}
"""

    class InteractionChoice(BaseModel):
        speaker_name: str
        listener_name: str

    system_prompt = "You are an expert in statistical modeling and cultural analysis."

    response = chat(
        model="deepseek-r1:7b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": get_user_prompt()}
        ],
        format=InteractionChoice.model_json_schema()
    )
    interaction_choice: InteractionChoice = InteractionChoice.model_validate_json(response.message.content)

    speaker = None
    listener = None
    if interaction_choice.speaker_name == agent_A.name:
        speaker = agent_A
    elif interaction_choice.speaker_name == agent_B.name:
        speaker = agent_B

    if interaction_choice.listener_name == agent_A.name:
        listener = agent_A
    elif interaction_choice.listener_name == agent_B.name:
        listener = agent_B

    return speaker, listener

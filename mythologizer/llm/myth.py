from typing import Union, Literal, List, Dict, Set
from pydantic import BaseModel, Field, conint, confloat, create_model, UUID4
from ollama import chat
import numpy as np

from mythologizer.agent import Agent
from mythologizer.culture import Culture, CultureRegistry
from mythologizer.myths import Myth


def ollame_get_myth_ratio(
        speaker: Agent,
        speaker_values: List[dict],
        listener: Agent,
        listener_values: List[dict],
        culture_registry: CultureRegistry
):
    def get_user_prompt():
        return f"""
You are an expert in statistical modeling and cultural analysis.

The following two agents just met and will tell each other a myth. Agent {speaker.id} with name {speaker.name} is telling the myth to {listener.id} with name {listener.name}.
Based on their attributes, values and well as cultures give me a ratio between 0 and 1. This valued will be used to specify how much of the original myth of the speaker will be used in combining it with the myth of the listener.

Speaker cultures: {[f"{culture.name}:{culture.description}" for culture in culture_registry[speaker.culture_ids]]}
Speaker values: {speaker_values}
----
Listeners cultures: {[f"{culture.name}:{culture.description}" for culture in culture_registry[listener.culture_ids]]}
Listener values: {listener_values}
"""

    class Ratio(BaseModel):
        ratio: float

    system_prompt = "You are an expert in statistical modeling and cultural analysis."

    response = chat(
        model="deepseek-r1:7b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": get_user_prompt()}
        ],
        format=Ratio.model_json_schema(),
    )
    interaction_choice: Ratio = Ratio.model_validate_json(response.message.content)
    return interaction_choice.ratio


def ollama_combine_myth(
        speaker: Agent,
        speaker_values: List[dict],
        speaker_myth: Myth,
        listener: Agent,
        listener_values: List[dict],
        listener_myth: Myth,
        combined_mythemes: Set[str],
        ratio: float,
        culture_registry: CultureRegistry
):
    def get_user_prompt():
        return f"""
You are an expert in statistical modeling and cultural analysis.

The following two agents just met and will tell each other a myth. Agent {speaker.id} with name {speaker.name} is telling the myth to {listener.id} with name {listener.name}.
Both myth are build from mythemes. Please combine both written out versions of the myth based on the combination ratio. Keep the style and nuance of both text in regards of the ratio. 
The combined myth should only reflect the given combined mythemes. Make sure of that. Also keep in mind that both agents have cultures that might influence the combined myth. 
The new combined myth will only be remembered my the listener. So his attributes and cultures influence the combination more. The ratio tells how much of the myth of the speaker should be used. For the listener it is 1-ratio. 

Speaker cultures: {[f"{culture.name}:{culture.description}" for culture in culture_registry[speaker.culture_ids]]}
Speaker values: {speaker_values}
Speaker mythemes: {speaker_myth.mythemes}
Speaker myth written out: {speaker_myth.current_myth}
----
Listeners cultures: {[f"{culture.name}:{culture.description}" for culture in culture_registry[listener.culture_ids]]}
Listener values: {listener_values}
Listener mythemes: {listener_myth.mythemes}
Listener myth written out: {listener_myth.current_myth}
----
Combined mythemes: {combined_mythemes}
Ratio: {ratio}
"""

    class CombinedMyth(BaseModel):
        myth: str

    system_prompt = "You are an expert in statistical modeling and cultural analysis."

    response = chat(
        model="deepseek-r1:7b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": get_user_prompt()}
        ],
        format=CombinedMyth.model_json_schema(),
    )
    combined_myth: CombinedMyth = CombinedMyth.model_validate_json(response.message.content)
    return combined_myth.myth


def ollama_mutate_myth(
        agent: Agent,
        agent_values: List[dict],
        myth: Myth,
        culture_registry: CultureRegistry,
        action: str
):
    def get_user_prompt():
        return f"""
You are an expert in statistical modeling and cultural analysis.

Agent {agent.id} with name {agent.name} has a myth that will be mutated with he following action: "{action}".
The myth is build from mythemes. Please apple the action "{action}" the one of the mythemes. Choose this mytheme at random.
After this update the written out version to reflect the change in mythemes. Keep the style and nuance of the original text. 
Also keep in mind that the agent has cultures aswell as attributes that might influence the myth.

Agent cultures: {[f"{culture.name}:{culture.description}" for culture in culture_registry[agent.culture_ids]]}
Agent values: {agent_values}
Mythemes: {myth.mythemes}
Myth Written out: {myth.current_myth}
action: {action}
"""

    class MutatedMyth(BaseModel):
        mythemes: List[str]
        written_out_myth: str

    system_prompt = "You are an expert in statistical modeling and cultural analysis."

    response = chat(
        model="deepseek-r1:7b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": get_user_prompt()}
        ],
        format=MutatedMyth.model_json_schema()
    )
    mutated_myth: MutatedMyth = MutatedMyth.model_validate_json(response.message.content)
    return mutated_myth.written_out_myth, set(mutated_myth.mythemes)

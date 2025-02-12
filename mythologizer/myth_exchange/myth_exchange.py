import logging
import random
from copy import deepcopy
from typing import Dict, Callable, Set, Tuple, Any
import numpy as np

from mythologizer.agent import Agent
from mythologizer.agent_attribute import AgentAttribute
from mythologizer.culture import Culture, CultureRegistry
from mythologizer.myths import Myth
from mythologizer.llm import (
    ollame_get_myth_ratio,
    ollama_combine_myth,
    ollama_mutate_myth,
)

# Configure logger for debugging.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def standard_remember_function(agent_attributes: Dict[str, float], length: int) -> int:
    """
    Selects an index from a range [0, length) based on the agent's "Recollection" and "Creativity" attributes.

    The selection is weighted by an exponentially decaying probability distribution whose scale is adjusted
    by the agent’s recollection (lower recollection steepens the decay) and creativity (adds a uniform component).

    Args:
        agent_attributes (Dict[str, float]): A dictionary of agent attributes. Must include the keys
            "Recollection" and "Creativity".
        length (int): The number of items among which to choose (must be positive).

    Returns:
        int: The selected index (0-indexed).

    Raises:
        ValueError: If `length` is not positive or if the attribute values are out of bounds.
        KeyError: If the required keys ("Recollection" or "Creativity") are missing.
    """
    if length <= 0:
        logger.error("Length must be positive, got %d", length)
        raise ValueError("Length must be positive.")

    if "Recollection" not in agent_attributes or "Creativity" not in agent_attributes:
        logger.error("Missing required keys in agent attributes: %s", agent_attributes)
        raise KeyError("Missing required attribute keys: 'Recollection' or 'Creativity'")

    recollection = agent_attributes["Recollection"]
    creativity = agent_attributes["Creativity"]

    if not (0 <= recollection <= 1):
        logger.error("Recollection attribute out of bounds: %f", recollection)
        raise ValueError("Recollection attribute must be between 0 and 1.")
    if not (0 <= creativity <= 1):
        logger.error("Creativity attribute out of bounds: %f", creativity)
        raise ValueError("Creativity attribute must be between 0 and 1.")

    scale_factor = (1 - recollection) * 20 + 1
    # Use indices 1..length to compute weights, then adjust back to 0-index.
    indices = np.arange(1, length + 1)
    probabilities = np.exp(-scale_factor * indices / length)
    probabilities = (1 - creativity) * probabilities + creativity * (1 / length)
    probabilities /= probabilities.sum()  # Normalize

    selected = int(np.random.choice(indices, p=probabilities))
    selected_index = selected - 1  # Adjust for 0-indexing
    logger.debug(
        "standard_remember_function: Selected index %d (raw value %d) with probabilities: %s",
        selected_index, selected, probabilities,
    )
    return selected_index


def tell_myth(
        culture_registry: CultureRegistry,
        listener_agent: Agent,
        listener_agent_values: Dict[str, float],
        speaker_agent: Agent,
        speaker_agent_values: Dict[str, float],
        index_sample_function: Callable[[Dict[str, float], int], int] = standard_remember_function,
) -> None:
    """
    Facilitates the process of one agent (speaker) telling a myth to another agent (listener).

    This function handles:
      - Recalling a myth from the speaker's memory.
      - Comparing the recalled myth with the myths in the listener's memory.
      - Combining the myths using a ratio determined via an LLM.
      - Optionally mutating myths based on their similarity.

    Args:
        culture_registry (CultureRegistry): A registry containing cultural context.
        listener_agent (Agent): The agent that listens to the myth.
        listener_agent_values (Dict[str, float]): Attributes for the listener agent.
        speaker_agent (Agent): The agent that tells the myth.
        speaker_agent_values (Dict[str, float]): Attributes for the speaker agent.
        index_sample_function (Callable): A function that selects an index from the speaker's memory.
            Defaults to `standard_remember_function`.

    Raises:
        ValueError: If the speaker's memory is empty.
    """
    logger.debug("tell_myth: Starting myth-telling process between speaker and listener.")

    if not speaker_agent.memory or len(speaker_agent.memory) == 0:
        logger.debug("Speaker agent memory is empty. Interaction aborted")
        return

    def combine_myths(myth_speaker: Myth, myth_listener: Myth, ratio: float) -> Tuple[Set[Any], str]:
        """
        Combines two myths (from the speaker and listener) using a given ratio.

        The combination is performed by selecting portions of each myth's themes and then calling
        an LLM function to generate a combined narrative.

        Args:
            myth_speaker (Myth): The speaker's myth.
            myth_listener (Myth): The listener's myth.
            ratio (float): A value between 0 and 1 indicating the contribution of the speaker's myth.

        Returns:
            Tuple[Set[Any], str]: A tuple containing the combined myth themes and the combined narrative.

        Raises:
            ValueError: If `ratio` is not between 0 and 1.
        """

        def get_percentage_of_set(mythemes: Set[Any], fraction: float) -> Set[Any]:
            """
            Samples a percentage of items from a set based on the given fraction.

            Args:
                mythemes (Set[Any]): The set of myth themes.
                fraction (float): Fraction of the set to sample (must be between 0 and 1).

            Returns:
                Set[Any]: A subset of myth themes.

            Raises:
                ValueError: If `fraction` is not between 0 and 1.
            """
            if not 0 <= fraction <= 1:
                logger.error("Fraction out of bounds: %f", fraction)
                raise ValueError("Fraction must be between 0 and 1.")
            sample_size = int(len(mythemes) * fraction)
            sample_size = min(sample_size, len(mythemes))
            sampled = set(random.sample(sorted(mythemes), sample_size)) if sample_size > 0 else set()
            logger.debug(
                "get_percentage_of_set: Sampling %d items out of %d with fraction %f resulting in: %s",
                sample_size, len(mythemes), fraction, sampled,
            )
            return sampled

        if not (0 <= ratio <= 1):
            logger.error("Ratio out of bounds: %f", ratio)
            raise ValueError("Ratio must be between 0 and 1.")

        mythemes_speaker = myth_speaker.mythemes
        mythemes_listener = myth_listener.mythemes

        combined_mythemes = get_percentage_of_set(mythemes_speaker, ratio) & get_percentage_of_set(mythemes_listener,
                                                                                                   1 - ratio)
        logger.debug("combine_myths: Combined myth themes: %s", combined_mythemes)

        combined_story = ollama_combine_myth(
            listener=listener_agent,
            listener_values=listener_agent_values,
            speaker=speaker_agent,
            speaker_values=speaker_agent_values,
            culture_registry=culture_registry,
            speaker_myth=myth_speaker,
            listener_myth=myth_listener,
            combined_mythemes=combined_mythemes,
            ratio=ratio,
        )
        logger.debug("combine_myths: Combined story: %s", combined_story)
        return combined_mythemes, combined_story

    # A delta value used to update myth retention levels.
    remember_and_listen_to_same_retention_delta = 0.3  # TODO: Test and fine-tune this value

    def remember_myth() -> Myth:
        """
        Recalls a myth from the speaker's memory based on the provided sampling function and
        increases its retention.

        Returns:
            Myth: The recalled myth from the speaker's memory.
        """
        index = index_sample_function(speaker_agent_values, len(speaker_agent.memory))
        logger.debug("remember_myth: Selected memory index %d", index)
        try:
            myth = speaker_agent.memory[index]
        except IndexError as e:
            logger.error(
                "Index %d is out of bounds for speaker memory of length %d", index, len(speaker_agent.memory)
            )
            raise e
        myth.retention += remember_and_listen_to_same_retention_delta
        logger.debug("remember_myth: Increased myth retention to %f", myth.retention)
        return myth

    told_myth = remember_myth()

    # If the listener has no memory, simply mutate a copy of the told myth and add it.
    if not listener_agent.memory or len(listener_agent.memory) == 0:
        logger.debug("tell_myth: Listener agent memory is empty; adding a mutated copy of the told myth.")
        copy_of_myth = deepcopy(told_myth)
        copy_of_myth.retention = 1.0
        mutate_myth(copy_of_myth, listener_agent, listener_agent_values, culture_registry)
        listener_agent.memory.add_myth(copy_of_myth)
        return

    # Compute similarity scores between the told myth and each myth in the listener's memory.
    similarities_to_listener_myths = np.array(
        [myth.compare_mythemes(told_myth) for myth in listener_agent.memory]
    )
    logger.debug("tell_myth: Similarities to listener myths: %s", similarities_to_listener_myths)

    most_similar_index = int(np.argmax(similarities_to_listener_myths))
    try:
        most_similar_myth = listener_agent.memory[most_similar_index]
    except (IndexError, AttributeError) as e:
        logger.error("Error accessing most similar myth at index %d: %s", most_similar_index, str(e))
        raise e

    similarity_most_similar = float(np.max(similarities_to_listener_myths))
    logger.debug("tell_myth: Most similar myth has similarity: %f", similarity_most_similar)

    if similarity_most_similar > 0.3:
        logger.debug("tell_myth: Sufficient similarity found (> 0.3); proceeding to combine myths.")
        speaker_listener_ratio = ollame_get_myth_ratio(
            culture_registry=culture_registry,
            listener_values=listener_agent_values,
            listener=listener_agent,
            speaker_values=speaker_agent_values,
            speaker=speaker_agent,
        )
        logger.debug("tell_myth: Obtained speaker-listener ratio: %f", speaker_listener_ratio)
        combined_mythemes, combined_story = combine_myths(told_myth, most_similar_myth, speaker_listener_ratio)

        if similarity_most_similar >= 0.5:
            logger.debug("tell_myth: High similarity (>= 0.5); updating the most similar myth.")
            most_similar_myth.mythemes = most_similar_myth.mythemes & combined_mythemes
            most_similar_myth.current_myth = combined_story
            most_similar_myth.retention += (remember_and_listen_to_same_retention_delta - 0.1)
        else:
            logger.debug("tell_myth: Moderate similarity; creating a new myth for the listener.")
            new_myth = Myth(current_myth=combined_story, mythemes=combined_mythemes)
            listener_agent.memory.add_myth(new_myth)
            mutate_myth(new_myth, listener_agent, listener_agent_values, culture_registry)
    else:
        logger.debug("tell_myth: No sufficiently similar myth found (similarity <= 0.3); mutating a copy.")
        copy_of_myth = deepcopy(told_myth)
        copy_of_myth.retention = 1.0
        mutate_myth(copy_of_myth, listener_agent, listener_agent_values, culture_registry)
        listener_agent.memory.add_myth(copy_of_myth)


def mutate_myth(
        myth: Myth,
        agent: Agent,
        agent_values: Dict[str, float],
        culture_registry: CultureRegistry,
) -> None:
    """
    Mutates a myth based on a randomly selected action.

    The function selects an action at random from {"leave", "mutate", "delete"}. If the chosen action
    is not "leave", it calls an LLM function to generate a mutated narrative and themes for the myth.

    Args:
        myth (Myth): The myth to potentially mutate.
        agent (Agent): The agent associated with the myth.
        agent_values (Dict[str, float]): The attributes of the agent.
        culture_registry (CultureRegistry): A registry containing cultural context.
    """
    logger.debug("mutate_myth: Starting mutation for myth with narrative: %s", myth.current_myth)
    actions = {"leave", "mutate", "delete"}
    random_action = random.choice(list(actions))
    logger.debug("mutate_myth: Randomly selected action: %s", random_action)

    if random_action != "leave":
        mutated_story, mutated_mythemes = ollama_mutate_myth(
            myth=myth,
            culture_registry=culture_registry,
            agent=agent,
            agent_values=agent_values,
            action=random_action,
        )
        logger.debug(
            "mutate_myth: Mutation result - new narrative: %s, new themes: %s",
            mutated_story,
            mutated_mythemes,
        )
        myth.current_myth = mutated_story
        myth.mythemes = set(mutated_mythemes)
    else:
        logger.debug("mutate_myth: 'leave' action selected; myth remains unchanged.")

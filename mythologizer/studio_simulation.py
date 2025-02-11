from mythologizer.culture import Culture, AttributeDistribution
from mythologizer.random_number_generator import RandomNumberGenerator as RNG
from mythologizer.llm import gtp4o_culture_agent_attribute_distribution_map, gtp4o_interaction_pair
from mythologizer.agent_attribute import AgentAttribute
from mythologizer.agent import Agent
from mythologizer.registry import Registry, KeyConfig
from mythologizer.population import Population
from mythologizer.agent_attribute_matrix import AgentAttributeMatrix
from mythologizer.population_handler import AgentLifecycleManager
from mythologizer.memory import Memory
from mythologizer.myths import Myth
from mythologizer.myth_exchange import tell_myth
import logging
from openai import OpenAI
from dotenv import load_dotenv
import uuid
import os
import numpy as np
import random

from typing import Any, List, Optional, Union, List, Tuple
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

if __name__ == "__main__":
    load_dotenv()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


    # --- Epoch Changing Functions ---

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


    # Culture

    dc = Culture(
        name="Design & Computation students",
        description="Members of this culture think transdisciplinary, research scientific topics "
                    "collaboratively and use artistic methods to present and communicate findings. They have "
                    "diverse academic backgrounds but are united by their love for creative exploration of "
                    "multi-dimensional research questions.")

    artist = Culture(
        name="Artist & Performers",
        description="Members of this culture are highly creative and have a strong urge to create. They "
                    "are characterized by out-of-box thinking and having innovative ideas. They have a "
                    "fascination for the history of art and study the masters that came before them.")

    activist = Culture(
        name="Activists",
        description="Members of this culture are highly active in political discourses and are "
                    "fighting to bring about political or social change. They are very secure in their "
                    "beliefs and share them loudly with the public.")

    geeks = Culture(
        name="Geeks",
        description="Members of this culture are knowledgeable about and obsessively interested in a "
                    "particular subject, especially one that is technical or of specialist or niche "
                    "interest. They engage in or discuss technical or computer-related tasks obsessively "
                    "and with great attention to detail.")
    # Countries:

    iranian = Culture(
        name="Iranian",
        description="Iranian culture is a rich tapestry of ancient traditions, poetry, and hospitality, "
                    "where deep-rooted artistic and intellectual heritage blends with a warm, "
                    "communal spirit that values generosity and storytelling. Iran’s culture celebrates "
                    "beauty, wisdom, and the power of human connection.")

    egyptian = Culture(
        name="Egyptian",
        description="Egyptian culture is a captivating fusion of ancient heritage and "
                    "vibrant modern traditions, where history, hospitality, "
                    "and a deep sense of community shape daily life. From the "
                    "awe-inspiring legacy of the pharaohs and mesmerizing Arabic "
                    "calligraphy to lively music, flavorful cuisine, "
                    "and warm gatherings, Egypt celebrates a rich and enduring "
                    "cultural identity.")

    turkish = Culture(
        name="Turkish",
        description="Turkish culture is a vibrant fusion of ancient traditions and modern influences, "
                    "where hospitality is a core value, and guests are treated with warmth and "
                    "generosity. With a deep appreciation for music, art, and cuisine, Turkish people "
                    "take pride in their heritage while embracing diversity and connection.")

    german = Culture(
        name="German",
        description="German culture is built on a strong foundation of efficiency, innovation, "
                    "and precision, while also valuing deep intellectual and artistic traditions. From "
                    "world-class engineering and philosophy to a love for nature and sustainability, "
                    "Germans balance hard work with a rich appreciation for community, culture, "
                    "and quality of life.")

    bosnian = Culture(
        name="Bosnian",
        description="Bosnian culture is a beautiful blend of Eastern and Western influences, shaped by a "
                    "history of resilience, hospitality, and strong community bonds. From heartfelt "
                    "sevdalinka music and rich coffee culture to the warmth of its people and the "
                    "tradition of welcoming guests with open arms, Bosnia celebrates a deep appreciation "
                    "for family, heritage, and togetherness.")

    american = Culture(
        name="American",
        description="American culture is a dynamic mix of diversity, innovation, and individuality, "
                    "where people from all backgrounds contribute to a constantly evolving society. "
                    "With a spirit of creativity, ambition, and openness, the U.S. thrives on cultural "
                    "expression—from music, film, and technology to a strong tradition of community, "
                    "optimism, and the pursuit of dreams.")

    bulgarian = Culture(
        name="Bulgarian",
        description="Bulgarian culture is a rich blend of ancient traditions, folklore, and warm "
                    "hospitality, where music, dance, and storytelling play a vital role in daily "
                    "life. From the breathtaking melodies of traditional bagpipes to vibrant "
                    "festivals and the deep appreciation for nature and family, Bulgaria embraces a "
                    "strong sense of heritage, resilience, and community.")

    canadian = Culture(
        name="Canadian",
        description="Canadian culture is defined by its inclusivity, diversity, and deep connection to "
                    "nature, where people value kindness, multiculturalism, and community spirit. From "
                    "breathtaking landscapes and Indigenous traditions to a strong appreciation for "
                    "the arts, hockey, and maple syrup, Canada embraces a harmonious blend of "
                    "heritage, progress, and warmth.")

    chinese = Culture(
        name="Chinese",
        description="Chinese culture is a profound blend of ancient wisdom and modern innovation, "
                    "deeply rooted in traditions of respect, family values, and artistic excellence. "
                    "From breathtaking calligraphy and cuisine to vibrant festivals like the Lunar New "
                    "Year, China embraces a rich heritage of philosophy, craftsmanship, and collective "
                    "harmony.")

    british = Culture(
        name="British",
        description="British culture is a unique blend of deep-rooted traditions and modern creativity, "
                    "where history, literature, and innovation thrive side by side. From the charm of "
                    "afternoon tea and historic landmarks to world-class music, humor, and a strong "
                    "sense of community, the UK embraces both heritage and forward-thinking spirit.")

    cultures = [dc, artist, activist, geeks, iranian, egyptian, turkish, german, bosnian, american, bulgarian, canadian,
                chinese, british]

    # attributes
    age = AgentAttribute(
        name='Age',
        description='Age of the agent',
        d_type=int,
        min=0,
        epoch_change_function=epoch_iterate
    )  # TODO iterate +1 epoch function

    confidence = AgentAttribute(
        name='Confidence',
        description='The confidence of the agent',
        d_type=float,
        min=0.0,
        max=1.0,
        epoch_change_function=epoch_random_fluctuation
    )

    emotionality = AgentAttribute(
        name='Emotionality',
        description='The emotionality of the agent with 0 representing a very emotionless person and 1 representing a very emotional person',
        d_type=float,
        min=0.0,
        max=1.0)

    creativity = AgentAttribute(
        name='Creativity',
        description='The creativity of the agent with 0 representing a non-creative person and 1 representing a very creative person',
        d_type=float,
        min=0.0,
        max=1.0)

    talkativeness = AgentAttribute(
        name='Talkativeness',
        description='The talkativeness of the agent with 0 representing a non-talkative person and 1 representing a very talkative person',
        d_type=float,
        min=0.0,
        max=1.0,
        epoch_change_function=epoch_random_fluctuation
    )

    popularity = AgentAttribute(
        name='Popularity',
        description='The popularity of the agent with 0 representing a not popular person and 1 representing a very popular person',
        d_type=float,
        min=0.0,
        max=1.0)  # sinus curve

    funiness = AgentAttribute(
        name='Funiness',
        description='The funniness of the agent with 0 representing an unfunny person and 1 representing a very funny person',
        d_type=float,
        min=0.0,
        max=1.0)  # constant

    loneliness = AgentAttribute(
        name='Loneliness',
        description='The loneliness of the agent with 0 representing a person who is more of a loner and 1 representing a very community-oriented person',
        d_type=float,
        min=0.0,
        max=1.0)  # constant

    stubbornness = AgentAttribute(
        name='Stubbornness',
        description='The stubbornness of the agent with 0 representing a very submissive person and 1 representing a very stubborn person',
        d_type=float,
        min=0.0,
        max=1.0)  # constant

    organization = AgentAttribute(
        name='Organization',
        description='The level of organization of the agent with 0 representing a very chaotic person and 1 representing a very organized person',
        d_type=float,
        min=0.0,
        max=1.0)  # constant

    superstitiousness = AgentAttribute(
        name='Superstitiousness',
        description='The level of superstition of the agent with 0 representing a very '
                    'rational person and 1 representing a very superstitious person',
        d_type=float,
        min=0.0,
        max=1.0)  # constant

    mysteriousness = AgentAttribute(
        name='Mysteriousness',
        description='The level of mysteriousness of the agent with 0 representing a '
                    'non-mysterious person and 1 representing a very mysterious person',
        d_type=float,
        min=0.0,
        max=1.0)  # constant

    clumsiness = AgentAttribute(
        name='Clumsiness',
        description='The level of clumsiness of the agent with 0 representing a well-handled '
                    'person and 1 representing a very clumsy person',
        d_type=float,
        min=0.0,
        max=1.0)  # constant

    calmness = AgentAttribute(
        name='Calmness',
        description='The level of calmness of the agent with 0 representing a hectic person and '
                    '1 representing a very calm person',
        d_type=float,
        min=0.0,
        max=1.0)  # constant

    smartness = AgentAttribute(
        name='Smartness',
        description='The level of smartness of the agent with 0 representing a slow person and '
                    '1 representing a very smart person',
        d_type=float,
        min=0.0,
        max=1.0)  # constant

    closeness_to_nature = AgentAttribute(
        name='Closeness to Nature',
        description='The level of how close an agent is to nature with 0 '
                    'representing an indoors person and 1 representing a very active '
                    'and outdoorsy person',
        d_type=float,
        min=0.0,
        max=1.0)  # constant

    tech_savvy = AgentAttribute(
        name='Tech Savvy',
        description='The level of tech-savviness of the agent with 0 representing a '
                    'non-technical person and 1 representing a very technology-affine person',
        d_type=float,
        min=0.0,
        max=1.0)  # constant

    craftiness = AgentAttribute(
        name='Craftiness',
        description='The level of craftiness of the agent with 0 representing a non-crafty '
                    'person and 1 representing a very crafty person',
        d_type=float,
        min=0.0,
        max=1.0)  # constant

    absurdity = AgentAttribute(
        name='Absurdity',
        description='The level of absurdity of the agent with 0 representing a logical person '
                    'and 1 representing a very absurd person',
        d_type=float,
        min=0.0,
        max=1.0,
        epoch_change_function=epoch_random_fluctuation
    )

    rebelliousness = AgentAttribute(
        name='Rebelliousness',
        description='The level of rebelliousness of the agent with 0 representing a '
                    'rule-following person and 1 representing a very rebellious person',
        d_type=float,
        min=0.0,
        max=1.0,
        epoch_change_function=epoch_random_fluctuation
    )

    recollection = AgentAttribute(
        name='Recollection',
        description='The level of recollection of the agent with 0 representing a very '
                    'forgetful person and 1 representing a person with a good memory',
        d_type=float,
        min=0.0,
        max=1.0,
        epoch_change_function=epoch_random_fluctuation
    )  # TODO make this worse with age

    attributes = [age, confidence, emotionality, creativity, talkativeness, popularity, funiness, loneliness,
                  stubbornness, organization, superstitiousness, mysteriousness, clumsiness, calmness, smartness,
                  closeness_to_nature, tech_savvy, craftiness, absurdity, rebelliousness, recollection]

    memory_size = 10

    # agents
    maryam = Agent(
        name="maryam",
        culture_ids={dc.id, geeks.id, iranian.id},
        myths=[
            Myth(
                current_myth='In a world where boundaries blurred, a hero named Orion stood at the edge of a cliff, seeking to breach the realm above. A mysterious figure, neither man nor beast, appeared beside him. "I am Liminal," it said, "I walk the lines between worlds." Liminal stretched out a hand, and a pillar of wind lifted Orion, raising him to the sky. There, Orion battled star-beasts and claimed a celestial map. But the sky refused to release him, its winds howling in defiance. Below, Liminal chanted, its voice echoing through the realms. The border between earth and sky shimmered, and Orion plummeted, the map clutched tight. Liminal caught him, their hands meeting at the threshold, and together, they stepped back onto solid ground. The map unfurled, revealing secrets to traverse the worlds, borderlines now mere paths for Orion to tread.',
                mythemes={"The hero", "Raises the hero to the sky", "Helps the hero return to earth",
                          "Helps the hero overcome borderlines between worlds"}
            )
        ]
    )
    att_values_maryam = [
        1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    lilli = Agent(name="lilli", culture_ids={dc.id, artist.id, german.id},
                  memory=Memory(
                      size=memory_size,
                      myths=[
                          Myth(
                              current_myth='The hero stood, arms outstretched, as the ancient crone raised her gnarled staff. With a cry, she struck the earth, and the hero was lifted, spiraling into the sky, swallowed by clouds. The crone whispered secrets to the wind, which carried the hero gently back to the ground. She pressed a stone into his palm, its surface swirling with worlds unseen. With her guidance, he stepped through borders once impervious, becoming a bridge between realms, a wanderer of worlds.',
                              mythemes={"The hero", "Raises the hero to the sky", "Helps the hero return to earth",
                                        "Helps the hero overcome borderlines between worlds"}
                          )
                      ]
                  ))
    att_values_lilli = [
        1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    sarah = Agent(name="sarah", culture_ids={dc.id, artist.id, activist.id, egyptian.id},
                  memory=Memory(
                      size=memory_size,
                      myths=[
                          Myth(
                              current_myth='The crone, ancient and bent, raised her staff. With a cry, she struck the earth, and the hero ascended, spiraling into the heavens. The crone\'s lips moved in silent incantation, her hands tracing patterns in the air. Gently, the hero descended, feet touching the ground, eyes wide with wonder. She handed him a stone, its surface shimmering with worlds unseen. "Step through," she whispered, and he did, boundaries dissolving, worlds merging at his touch.',
                              mythemes={"The crone", "Raises the hero to the sky", "Helps the hero return to earth",
                                        "Helps the hero overcome borderlines between worlds"}
                          )
                      ]
                  ))
    att_values_sarah = [
        1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    harun = Agent(name="harun", culture_ids={dc.id, geeks.id, bosnian.id},
                  memory=Memory(
                      size=memory_size,
                      myths=[
                          Myth(
                              current_myth='The crone, eyes like embers, leaned close. "Eternal life," she whispered, "if you do my bidding, against your heart\'s cry." The hero hesitated, then nodded. She cackled, pressing a stone into his palm. "Step through," she urged, pointing to a shimmering boundary. He did, gasping as worlds shifted around him. With each border crossed, his convictions frayed, life stretching endless before him, a bittersweet promise kept.',
                              mythemes={"The crone",
                                        "Promises the hero eternal life if he goes against his convictions and does her a favor",
                                        "Helps the hero overcome borderlines between worlds"}
                          )
                      ]
                  ))
    att_values_harun = [
        1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    sena = Agent(name="sena", culture_ids={dc.id, artist.id, activist.id, turkish.id},
                 memory=Memory(
                     size=memory_size,
                     myths=[
                         Myth(
                             current_myth='The crone, eyes like embers, leaned close. "Eternal life," she whispered, "if you do my bidding, against your heart\'s cry." The hero hesitated, then nodded. She cackled, pressing a stone into his palm. "Step through," she urged, pointing to a shimmering boundary. He did, gasping as worlds shifted around him. With each border crossed, his convictions frayed, life stretching endless before him, a bittersweet promise kept.',
                             mythemes={"The crone",
                                       "Promises the hero eternal life if he goes against his convictions and does her a favor",
                                       "The crone turns into a beast that leads the hero through his quest",
                                       "His convictions are strenghtened.",
                                       "Helps the hero overcome borderlines between worlds"}
                         )
                     ]
                 ))
    att_values_sena = [
        1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    finn = Agent(name="finn", culture_ids={dc.id, geeks.id, american.id},
                 memory=Memory(
                     size=memory_size,
                     myths=[
                         Myth(
                             current_myth='In the heart of a withered forest, the crone, skin like ancient bark, offered a promise. "Eternal life, if you betray your honor and do my bidding." The hero, steel in his heart, agreed. The crone twisted, bones cracking, flesh warping, until a beast stood before him, eyes burning like embers. It led him through shadows, across rivers of sorrow, and into the veil between worlds. With each step, the hero\'s convictions frayed, but the beast\'s strength was his guide, tearing through the borderlines of reality, forging a path towards immortality.',
                             mythemes={"The crone",
                                       "Promises the hero eternal life if he goes against his convictions and does her a favor",
                                       "The crone turns into a beast that leads the hero through his quest",
                                       "Helps the hero overcome borderlines between worlds"}
                         )
                     ]
                 ))
    att_values_finn = [
        1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    engy = Agent(name="engy", culture_ids={dc.id, artist.id, egyptian.id, german.id},
                 memory=Memory(
                     size=memory_size,
                     myths=[
                         Myth(
                             current_myth='In the heart of a withered forest, the crone, skin like ancient bark, offered a promise. "Eternal life, if you betray your honor and do my bidding." The hero, steel in his heart, agreed. The crone twisted, bones cracking, flesh warping, until a beast stood before him, eyes burning like embers. It led him through shadows, across rivers of sorrow, and into the veil between worlds. With each step, the hero\'s convictions frayed, but the beast\'s strength was his guide, tearing through the borderlines of reality, forging a path towards immortality.',
                             mythemes={"The crone",
                                       "Promises the hero eternal life if he goes against his convictions and does her a favor",
                                       "The crone turns into a beast that leads the hero through his quest",
                                       "Helps the hero overcome borderlines between worlds"}
                         )
                     ]
                 ))
    att_values_engy = [
        1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    paul = Agent(name="paul", culture_ids={dc.id, german.id},
                 memory=Memory(
                     size=memory_size,
                     myths=[
                         Myth(
                             current_myth='In the heart of a whispering forest, a crone, skin like ancient bark, offered a promise. "Eternal life, if you bend your virtues, do my bidding." The hero, eyes ablaze with resolve, agreed. The crone twisted, bones cracking, flesh warping, becoming a beast of shadow and sinew. It led him through mires and madness, his quest a echoing nightmare. Victory in hand, the beast turned, eyes burning like embers. "Eternity, as promised," it growled, forcing the hero into the underworld\'s maw, to live forever in the dark.',
                             mythemes={"The crone",
                                       "Promises the hero eternal life if he goes against his convictions and does her a favor",
                                       "The crone turns into a beast that leads the hero through his quest",
                                       "The crone forces the hero to live in the underworld for eternity"}
                         )
                     ]
                 ))
    att_values_paul = [
        1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    lora = Agent(name="lora", culture_ids={dc.id, bulgarian.id, geeks.id})
    att_values_lora = [
        1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    sam = Agent(name="sam", culture_ids={dc.id, artist.id, canadian.id})
    att_values_sam = [
        1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    ben = Agent(name="ben", culture_ids={dc.id, geeks.id, german.id})
    att_values_ben = [
        1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    kenny = Agent(name="kenny", culture_ids={dc.id, artist.id, activist.id, chinese.id})
    att_values_kenny = [
        1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    max = Agent(name="max", culture_ids={dc.id, artist.id, geeks.id, british.id})
    att_values_max = [
        1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    agents = [maryam, lilli, sarah, harun, sena, finn, engy, paul, lora, sam, ben, kenny, max]
    attribute_values = [att_values_maryam, att_values_lilli, att_values_sarah, att_values_harun, att_values_sena,
                        att_values_finn, att_values_engy, att_values_paul, att_values_lora, att_values_sam,
                        att_values_ben, att_values_kenny, att_values_max]

    for att in attribute_values:
        if len(att) != len(attributes):
            raise ValueError(
                f"agent attributes {len(att)} must be the same length as attribute definitions {len(attributes)}")

    agent_lifecycle_manager = AgentLifecycleManager(
        agent_attributes=attributes,
        agents=agents,
        cultures=cultures,
        attribute_values=attribute_values
    )


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

    while current_epoch < 10:

        agent_lifecycle_manager.agent_attribute_matrix.apply_epoch_changing_functions()

        pairs = get_random_interaction_tuples(number_interactions, agent_lifecycle_manager.population)
        for pair in pairs:
            agent_a, agent_b = pair
            agent_a_values = agent_lifecycle_manager.agent_attribute_matrix.agent_attribute_register.create_values_dict(
                agent_lifecycle_manager.agent_attribute_matrix.matrix[agent_a.index])
            agent_b_values = agent_lifecycle_manager.agent_attribute_matrix.agent_attribute_register.create_values_dict(
                agent_lifecycle_manager.agent_attribute_matrix.matrix[agent_b.index])

            speaker, listener = gtp4o_interaction_pair(open_ai_client=openai_client, agent_A=agent_a,
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

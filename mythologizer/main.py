from mythologizer.culture import Culture, CultureRegister, AttributeDistribution, AttributesDistributions
from mythologizer.random_number_generator import RandomNumberGenerator as RNG
from mythologizer.llm import gtp4o_culture_agent_attribute_distribution_map
import logging
from openai import OpenAI
from dotenv import load_dotenv
import os




if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    rng = RNG(1)
    openai_client = OpenAI(api_key=os.getenv("OPENAI_KEY"))


    att_dists = gtp4o_culture_agent_attribute_distribution_map(open_ai_client=openai_client,
                                                               distribution_map=rng.distributions_map,
                                                               agent_attribute_names=["happy", "evil", "helpfulness", "talkative"],
                                                               culture_name="Witches",
                                                               culture_description="Witches are a mystical culture with deep connections to nature and magic. They value wisdom, secrecy, and the balance of natural forces.")

    print(att_dists)

                                                                            




    """
    # sample = rng.distributions["beta"].sample({"a": 1, "b": 0.5}, size=10)

    happy_dist = AttributeDistribution(name="happy", distribution=rng.distributions["beta"],
                                       parameters={"a": 1, "b": 0.5})
    swag_dist = AttributeDistribution(name="swaggy", distribution=rng.distributions["beta"],
                                      parameters={"a": 2, "b": 0.3})

    att_dists = AttributesDistributions(attributes_distributions=[happy_dist, swag_dist])
    #print(att_dists)
    """


    """
    tech_bros = Culture(name="Tech bros", description="naughty boys")
    haexxen = Culture(name="Häxxen", description="naughty witches")

    culture_register = CultureRegister(cultures=[tech_bros])
    culture_register.add_culture(haexxen)
    """

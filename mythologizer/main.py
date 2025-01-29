from mythologizer.culture import Culture, CultureRegister, AttributeDistribution, AttributesDistributions
from mythologizer.random_number_generator import RandomNumberGenerator as RNG
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    rng = RNG(1)
    # sample = rng.distributions["beta"].sample({"a": 1, "b": 0.5}, size=10)

    happy_dist = AttributeDistribution(name="happy", distribution=rng.distributions["beta"],
                                       parameters={"a": 1, "b": 0.5})
    swag_dist = AttributeDistribution(name="swaggy", distribution=rng.distributions["beta"],
                                      parameters={"a": 2, "b": 0.3})

    att_dists = AttributesDistributions(attributes_distributions=[happy_dist, swag_dist])
    print(att_dists.sample(3))
    """
    tech_bros = Culture(name="Tech bros", description="naughty boys")
    haexxen = Culture(name="Häxxen", description="naughty witches")

    culture_register = CultureRegister(cultures=[tech_bros])
    culture_register.add_culture(haexxen)
    """

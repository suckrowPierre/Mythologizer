from mythologizer.culture import Culture, CultureRegister
from mythologizer.random_number_generator import RandomNumberGenerator as RNG
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    rng = RNG(1)
    sample = rng.sample_from_distribution("beta", {"a": 1, "b": 0.5}, size=10)
    print(sample)



    """
    tech_bros = Culture(name="Tech bros", description="naughty boys")
    haexxen = Culture(name="Häxxen", description="naughty witches")

    culture_register = CultureRegister(cultures=[tech_bros])
    culture_register.add_culture(haexxen)
    """
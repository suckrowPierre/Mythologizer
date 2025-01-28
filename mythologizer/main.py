from mythologizer.culture import Culture, CultureRegister
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    tech_bros = Culture(name="Tech bros", description="naughty boys")
    haexxen = Culture(name="Häxxen", description="naughty witches")

    culture_register = CultureRegister(cultures=[tech_bros])
    culture_register.add_culture(haexxen)
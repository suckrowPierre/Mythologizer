import math
import numpy as np


def get_culture_indicies(n_agents, number_of_cultures):
    agents_per_culture = math.floor(n_agents / number_of_cultures)
    range_cultures = range(number_of_cultures)
    culture_indices = [[i] * agents_per_culture for i in range_cultures]
    culture_indices = np.concatenate(np.array(culture_indices)).tolist()
    for _ in range(n_agents - len(culture_indices)):
        culture_indices.append(range_cultures[-1])

    return culture_indices


def main():
    print(get_culture_indicies(10, 1))


if __name__ == '__main__':
    main()

@validate_call
def get_cultures_random_with_overlap(
        rng: RNG,
        n_agents: StrictInt = Field(..., ge=1, description="Must be a positive integer >= 1"),
        n_of_cultures: StrictInt = Field(..., ge=1, description="Must be a positive integer >= 1"),
        probability_to_be_in_one_culture: Optional[float] = Field(
            default=None,
            ge=0.0,
            le=1.0,
            description="Must be a value between 0 and 1",
        ),
        **kwargs  # Accept additional keyword arguments
) -> List[Tuple[bool, ...]]:
    """
    Generates a list of tuples indicating culture memberships for agents.

    Each tuple corresponds to an agent, and each boolean in the tuple indicates
    whether the agent is part of a particular culture.

    If no culture is assigned to an agent, the function ensures that the agent
    is assigned to at least one culture randomly.

    Args:
        rng (RNG): random number generator
        n_agents (StrictInt): Number of agents (must be >= 1).
        n_of_cultures (StrictInt): Number of cultures (must be >= 1).
        probability_to_be_in_one_culture (Optional[float]):
            Probability for an agent to be part of a culture
            (between 0 and 1). Defaults to 1 / n_of_cultures if None.

    Returns:
        List[Tuple[bool, ...]]: List of tuples indicating culture memberships.
    """
    if probability_to_be_in_one_culture is None:
        probability_to_be_in_one_culture = 1 / n_of_cultures
        logger.debug(
            f"Probability not provided. Set to default 1/{n_of_cultures} = {probability_to_be_in_one_culture}"
        )
    logger.debug(
        f"Creating culture indices for {n_agents} agents across {n_of_cultures} cultures with overlap."
    )
    arr = rng.numpy_rng.rand(n_agents, n_of_cultures) < probability_to_be_in_one_culture
    all_false = ~arr.any(axis=1)
    arr[all_false, np.random.randint(n_of_cultures, size=all_false.sum())] = True
    return list(map(tuple, arr))


@validate_call
def get_cultures_uniform_no_overlap(
        n_agents: StrictInt = Field(..., ge=1, description="Must be a positive integer >= 1"),
        n_of_cultures: StrictInt = Field(..., ge=1, description="Must be a positive integer >= 1"),
        **kwargs  # Accept additional keyword arguments
) -> List[Tuple[bool, ...]]:
    """
    Distribute agents uniformly across cultures with no overlapping memberships.

    This function assigns each agent to exactly one culture. The agents are
    distributed as evenly as possible among the available cultures. If the number
    of agents isn't perfectly divisible by the number of cultures, the remaining
    agents are assigned to the first few cultures to maintain uniformity.

    Each agent's cultural membership is represented as a tuple of boolean values,
    where each position corresponds to a specific culture. A `True` value indicates
    membership in that culture, while `False` denotes non-membership. This ensures
    that each agent belongs to one and only one culture, eliminating any overlap.

    Args:
        n_agents (StrictInt):
            Total number of agents. Must be a positive integer (>= 1).
        n_of_cultures (StrictInt):
            Number of distinct cultures. Must be a positive integer (>= 1).

    Returns:
        List[Tuple[bool, ...]]: List of tuples indicating culture memberships.
        """
    base, rem = divmod(n_agents, n_of_cultures)
    return [
        tuple(1 if j == i else 0 for j in range(n_of_cultures))
        for i in range(n_of_cultures)
        for _ in range(base + (i < rem))
    ]
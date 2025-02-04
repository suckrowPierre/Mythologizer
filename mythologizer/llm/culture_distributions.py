from typing import Union, Literal, List, Dict
from pydantic import BaseModel, Field, conint, confloat, create_model
from openai import OpenAI
from mythologizer.random_number_generator import ProbabilityDistributionMap
from mythologizer.culture import AttributesDistributions, AttributeDistribution


def gtp4o_culture_agent_attribute_distribution_map(
        open_ai_client: OpenAI,
        culture_name: str,
        culture_description: str,
        distribution_map: ProbabilityDistributionMap,
        agent_attribute_names: List[str],
        batch_size=10,
        n_max_retries=5,
):
    # TODO later refactor agent_attribute_names to list of agent_attribute to use validation for max and min value (in prompt and in the end)
    # TODO implement retries
    def create_distribution_choice(distribution_list: List[Dict[str, List[str]]]):
        parameter_classes = []
        distribution_names = []

        for dist in distribution_list:
            name = dist['name']
            class_name = name.capitalize()
            parameters = dist['parameters']
            distribution_names.append(name)

            fields = {param: (float, ...) for param in parameters}

            cls = create_model(class_name, **fields)
            parameter_classes.append(cls)

        name_literal = Literal[tuple(distribution_names)]
        parameters_union = Union[tuple(parameter_classes)]

        DistributionChoice = create_model(
            'DistributionChoice',
            name=(name_literal, ...),
            parameters=(parameters_union, ...)
        )

        return create_model(
            'DistributionChoices',
            choices=(List[DistributionChoice], ...)
        )

    def batch_agent_attributes(strings, n):
        return [strings[i:i + n] for i in range(0, len(strings), n)]

    def get_user_prompt(attribute_list):
        return f"""
You are an expert in statistical modeling and cultural analysis.

Given the following attributes:
{attribute_list}

Possible probability distributions:
{str(distribution_map.distributions)}

Culture Description:
{culture_description}

Culture Name:
{culture_name}

Assign to each attribute a probability distribution from the possible distributions list. In a list of length {len(attribute_list)}, provide concrete values for the distribution parameters that are appropriate for the given culture. Ensure that the parameters satisfy their validation rules. 
"""

    batches = batch_agent_attributes(agent_attribute_names, batch_size)

    attribute_distributions = []
    distribution_choice = create_distribution_choice(distribution_map.get_dict_list())
    system_prompt = "You are an expert in statistical modeling and cultural analysis."

    for batch in batches:
        #TODO: maybe add seed
        response = open_ai_client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": get_user_prompt(attribute_list=batch)}
            ],
            response_format=distribution_choice
        )
        distributions = response.choices[0].message.parsed.model_dump()["choices"]
        if len(distributions) != len(batch):
            raise Exception("Response list not same length as batch")

        for attribute_name, distribution in zip(batch, distributions):
            attribute_distributions.append(
                AttributeDistribution(name=attribute_name, distribution=distribution_map[distribution["name"]],
                                      parameters=distribution["parameters"]))

    return AttributesDistributions(attributes_distributions=attribute_distributions)

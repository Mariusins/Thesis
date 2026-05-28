"""Postprocess LLM call helpers, migrated to openai>=1.x.

Exports:
    prompt_chatgpt(system_input, user_input, temperature, save_path, index, ...)
    build_plan_format_conversion_prompt(...)
    build_query_generation_prompt(data)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Iterable, List, TypeVar

from datasets import load_dataset
from openai import OpenAI, APIError, APIConnectionError, RateLimitError, AuthenticationError, BadRequestError
from tqdm import tqdm
import func_timeout
from func_timeout import func_set_timeout


T = TypeVar("T")

_API_KEY = os.environ.get("OPENAI_API_KEY")
_CLIENT = OpenAI(api_key=_API_KEY) if _API_KEY else None


class TimeoutError(Exception):
    pass


def _client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        _CLIENT = OpenAI(api_key=key)
    return _CLIENT


@func_set_timeout(120)
def _chat_completion(model: str, messages: list, temperature: float, max_tokens: int = 2048):
    return _client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def batchify(data: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    assert batch_size > 0
    batch = []
    for item in data:
        if len(batch) == batch_size:
            yield batch
            batch = []
        batch.append(item)
    if batch:
        yield batch


def openai_unit_price(model_name, token_type="prompt"):
    if "gpt-4o-mini" in model_name:
        return 0.00015 if token_type == "prompt" else 0.0006
    if "gpt-4" in model_name:
        return 0.03 if token_type == "prompt" else 0.06
    if "gpt-3.5-turbo" in model_name:
        return 0.002
    return -1


def calc_cost_w_tokens(total_tokens: int, model_name: str) -> float:
    unit = openai_unit_price(model_name, token_type="completion")
    return round(unit * total_tokens / 1000, 6)


def calc_cost_w_prompt(total_tokens: int, model_name: str) -> float:
    unit = openai_unit_price(model_name, token_type="prompt")
    return round(unit * total_tokens / 1000, 6)


def get_perplexity(logprobs):
    assert len(logprobs) > 0, logprobs
    return math.exp(-sum(logprobs) / len(logprobs))


def prompt_chatgpt(system_input, user_input, temperature, save_path, index, history=None, model_name="gpt-4o-mini"):
    """Single-turn chat completion. Appends '<index>\\t<flattened-output>\\n' to save_path."""
    if history is None:
        history = [{"role": "system", "content": system_input}]
    history.append({"role": "user", "content": user_input})

    backoff = 2.0
    completion = None
    for _ in range(8):
        try:
            completion = _chat_completion(
                model=model_name,
                messages=history,
                temperature=temperature,
            )
            break
        except func_timeout.exceptions.FunctionTimedOut:
            print("Timeout, retrying...")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
        except RateLimitError:
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
        except (APIConnectionError, APIError):
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
        except (AuthenticationError, BadRequestError):
            raise

    if completion is None:
        raise RuntimeError("prompt_chatgpt: exhausted retries")

    assistant_output = completion.choices[0].message.content or ""
    history.append({"role": "assistant", "content": assistant_output})
    total_prompt_tokens = completion.usage.prompt_tokens
    total_completion_tokens = completion.usage.completion_tokens

    with open(save_path, "a+", encoding="utf-8") as f:
        flattened = str(index) + "\t" + "\t".join(x for x in assistant_output.split("\n"))
        f.write(flattened + "\n")

    cost = calc_cost_w_prompt(total_prompt_tokens, model_name) + calc_cost_w_tokens(total_completion_tokens, model_name)
    return assistant_output, history, cost


def build_query_generation_prompt(data):
    prompt_list = []
    prefix = """Given a JSON, please help me generate a natural language query. In the JSON, 'org' denotes the departure city. When 'days' exceeds 3, 'visiting_city_number' specifies the number of cities to be covered in the destination state. Please disregard the 'level' attribute. Here are three examples.

-----EXAMPLE 1-----
JSON:
{"org": "Gulfport", "dest": "Charlotte", "days": 3, "visiting_city_number": 1, "date": ["2022-03-05", "2022-03-06", "2022-03-07"], "people_number": 1, "local_constraint": {"house rule": null, "cuisine": null, "room type": null}, "budget": 1800, "query": null, "level": "easy"}
QUERY:
Please design a travel plan departing Gulfport and heading to Charlotte for 3 days, spanning March 5th to March 7th, 2022, with a budget of $1800.
-----EXAMPLE 2-----
JSON:
{"org": "Omaha", "dest": "Colorado", "days": 5, "visiting_city_number": 2, "date": ["2022-03-14", "2022-03-15", "2022-03-16", "2022-03-17", "2022-03-18"], "people_number": 7, "local_constraint": {"house rule": "pets", "cuisine": null, "room type": null}, "budget": 35300, "query": null, "level": "medium"}
QUERY:
Could you provide a  5-day travel itinerary for a group of 7, starting in Omaha and exploring 2 cities in Colorado between March 14th and March 18th, 2022? Our budget is set at $35,300, and it's essential that our accommodations be pet-friendly since we're bringing our pets.
-----EXAMPLE 3-----
JSON:
{"org": "Indianapolis", "dest": "Georgia", "days": 7, "visiting_city_number": 3, "date": ["2022-03-01", "2022-03-02", "2022-03-03", "2022-03-04", "2022-03-05", "2022-03-06", "2022-03-07"], "people_number": 2, "local_constraint": {"flight time": null, "house rule": null, "cuisine": ["Bakery", "Indian"], "room type": "entire room", "transportation": "self driving"}, "budget": 6200, "query": null, "level": "hard"}
QUERY:
I'm looking for a week-long travel itinerary for 2 individuals. Our journey starts in Indianapolis, and we intend to explore 3 distinct cities in Georgia from March 1st to March 7th, 2022. Our budget is capped at $6,200. For our accommodations, we'd prefer an entire room. We plan to navigate our journey via self-driving. In terms of food, we're enthusiasts of bakery items, and we'd also appreciate indulging in genuine Indian cuisine.

JSON\n"""
    for unit in data:
        unit = str(unit).replace(", 'level': 'easy'", '').replace(", 'level': 'medium'", '').replace(", 'level': 'hard'", '')
        prompt = prefix + str(unit) + "\nQUERY\n"
        prompt_list.append(prompt)
    return prompt_list


def build_plan_format_conversion_prompt(directory, set_type='validation', model_name='gpt4', strategy='direct', mode='two-stage'):
    prompt_list = []
    prefix = """Please assist me in extracting valid information from a given natural language text and reconstructing it in JSON format, as demonstrated in the following example. If transportation details indicate a journey from one city to another (e.g., from A to B), the 'current_city' should be updated to the destination city (in this case, B). Use a ';' to separate different attractions, with each attraction formatted as 'Name, City'. If there's information about transportation, ensure that the 'current_city' aligns with the destination mentioned in the transportation details (i.e., the current city should follow the format 'from A to B'). Also, ensure that all flight numbers and costs are followed by a colon (i.e., 'Flight Number:' and 'Cost:'), consistent with the provided example. Each item should include ['day', 'current_city', 'transportation', 'breakfast', 'attraction', 'lunch', 'dinner', 'accommodation']. Replace non-specific information like 'eat at home/on the road' with '-'. Additionally, delete any '$' symbols.
-----EXAMPLE-----
 [{{
        "days": 1,
        "current_city": "from Dallas to Peoria",
        "transportation": "Flight Number: 4044830, from Dallas to Peoria, Departure Time: 13:10, Arrival Time: 15:01",
        "breakfast": "-",
        "attraction": "Peoria Historical Society, Peoria;Peoria Holocaust Memorial, Peoria;",
        "lunch": "-",
        "dinner": "Tandoor Ka Zaika, Peoria",
        "accommodation": "Bushwick Music Mansion, Peoria"
    }},
    {{
        "days": 2,
        "current_city": "Peoria",
        "transportation": "-",
        "breakfast": "Tandoor Ka Zaika, Peoria",
        "attraction": "Peoria Riverfront Park, Peoria;The Peoria PlayHouse, Peoria;Glen Oak Park, Peoria;",
        "lunch": "Cafe Hashtag LoL, Peoria",
        "dinner": "The Curzon Room - Maidens Hotel, Peoria",
        "accommodation": "Bushwick Music Mansion, Peoria"
    }},
    {{
        "days": 3,
        "current_city": "from Peoria to Dallas",
        "transportation": "Flight Number: 4045904, from Peoria to Dallas, Departure Time: 07:09, Arrival Time: 09:20",
        "breakfast": "-",
        "attraction": "-",
        "lunch": "-",
        "dinner": "-",
        "accommodation": "-"
    }}]
-----EXAMPLE END-----
"""
    if set_type == 'train':
        query_data_list = load_dataset('osunlp/TravelPlanner', 'train')['train']
    elif set_type == 'validation':
        query_data_list = load_dataset('osunlp/TravelPlanner', 'validation')['validation']
    elif set_type == 'test':
        query_data_list = load_dataset('osunlp/TravelPlanner', 'test')['test']

    idx_number_list = list(range(1, len(query_data_list) + 1))
    if mode == 'two-stage':
        suffix = ''
    elif mode == 'sole-planning':
        suffix = f'_{strategy}'
    for idx in tqdm(idx_number_list):
        generated_plan = json.load(open(f'{directory}/{set_type}/generated_plan_{idx}.json'))
        key = f'{model_name}{suffix}_{mode}_results'
        if generated_plan[-1].get(key):
            prompt = prefix + "Text:\n" + generated_plan[-1][key] + "\nJSON:\n"
        else:
            prompt = ""
        prompt_list.append(prompt)
    return prompt_list

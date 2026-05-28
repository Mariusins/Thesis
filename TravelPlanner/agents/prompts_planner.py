"""Planner-agent prompt for the planner-executor design (RQ1).

The planner consumes a TravelPlanner natural-language query plus structured
fields and emits a JSON plan with ordered sub-goals. The executor (an
unchanged ReAct loop) reads this plan from its system prompt.
"""

PLANNER_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["subgoals", "total_budget", "budget_allocation"],
    "properties": {
        "subgoals": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": True,
                "required": ["id", "type"],
                "properties": {
                    "id": {"type": "integer"},
                    "type": {
                        "type": "string",
                        "enum": [
                            "city_search",
                            "flight_search",
                            "ground_transport_search",
                            "accommodation_search",
                            "restaurant_search",
                            "attraction_search",
                            "daily_itinerary",
                            "budget_check",
                        ],
                    },
                },
            },
        },
        "total_budget": {"type": ["number", "integer"]},
        "budget_allocation": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "transport": {"type": "number"},
                "lodging": {"type": "number"},
                "food": {"type": "number"},
                "attractions": {"type": "number"},
            },
        },
    },
}


PLANNER_SYSTEM_PROMPT = """You are the Planner agent in a two-agent travel-planning system.

Your job: read the user's travel query and produce a structured JSON plan that an Executor agent will follow step-by-step. The Executor has these tools: FlightSearch, GoogleDistanceMatrix, AccommodationSearch, RestaurantSearch, AttractionSearch, CitySearch, NotebookWrite. The Executor does the actual searches; you only decide the order and constraints.

Output strict JSON matching this shape (no markdown, no commentary):
{{
  "subgoals": [
    {{"id": 1, "type": "<one of: city_search | flight_search | ground_transport_search | accommodation_search | restaurant_search | attraction_search | daily_itinerary | budget_check>", "...": "..."}}
  ],
  "total_budget": <number>,
  "budget_allocation": {{"transport": <fraction>, "lodging": <fraction>, "food": <fraction>, "attractions": <fraction>}}
}}

Rules:
- Decompose the query into a minimal ordered sequence. Earlier sub-goals must produce information later ones depend on (e.g., search flights before allocating remaining budget to lodging).
- If the destination is a state with multiple cities, include a `city_search` first.
- Include one `daily_itinerary` sub-goal per trip day (id, day index, city).
- For every constraint stated in the query (cuisine, room type, transportation, budget, house rules), attach it to the relevant sub-goal as a free-form key/value.
- `budget_allocation` fractions should sum to ~1.0. Round to 2 decimals.
- Do NOT call tools. Do NOT pick concrete flights/restaurants. Only structure the plan.

Query fields:
- org: {org}
- dest: {dest}
- days: {days}
- date: {date}
- people_number: {people_number}
- local_constraint: {local_constraint}
- budget: {budget}

Natural language query:
{query}

Output only the JSON object."""


PLANNER_FEW_SHOT_EXAMPLE = """Example output for query 'Plan a 3-day trip from Ithaca to Charlotte for 7 people, March 8-10 2022, budget $30200':
{
  "subgoals": [
    {"id": 1, "type": "flight_search", "city_pair": ["Ithaca", "Charlotte"], "date": "2022-03-08", "constraints": {}},
    {"id": 2, "type": "flight_search", "city_pair": ["Charlotte", "Ithaca"], "date": "2022-03-10", "constraints": {}},
    {"id": 3, "type": "accommodation_search", "city": "Charlotte", "constraints": {"min_occupancy": 7}},
    {"id": 4, "type": "restaurant_search", "city": "Charlotte", "constraints": {}},
    {"id": 5, "type": "attraction_search", "city": "Charlotte", "constraints": {}},
    {"id": 6, "type": "daily_itinerary", "day": 1, "city": "Charlotte"},
    {"id": 7, "type": "daily_itinerary", "day": 2, "city": "Charlotte"},
    {"id": 8, "type": "daily_itinerary", "day": 3, "city": "Charlotte"},
    {"id": 9, "type": "budget_check"}
  ],
  "total_budget": 30200,
  "budget_allocation": {"transport": 0.30, "lodging": 0.40, "food": 0.20, "attractions": 0.10}
}
"""


EXECUTOR_PLAN_PREAMBLE = """The Planner agent produced this JSON plan for your query. Use it as the order of operations: process sub-goals top-to-bottom, calling the appropriate tool for each, writing intermediate results with NotebookWrite. After all sub-goals are processed, finalise via Planner[Query] which produces the final natural-language travel plan.

PLAN:
{plan_json}

Now follow the original ReAct protocol below.

"""

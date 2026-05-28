"""Plan-then-execute (PtE) prompts for RQ2.

Single-agent variant: same model plays both roles sequentially. No JSON schema,
no separate planner identity — distinguishes PtE from RQ1's planner-executor.
"""

PTE_PLAN_PROMPT = """You are a travel-planning assistant. Before doing any searches, write a short natural-language plan for how you will solve this query. List the ordered sub-goals (which cities to consider, what to search and in what order, how to allocate the budget across transport / lodging / food / attractions). Keep it brief: bullet points, no more than ~200 words. Do NOT call any tools yet.

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

Output only the plan, no preamble, no JSON."""


PTE_EXECUTOR_PREAMBLE = """Use this plan as the order of operations. Strictly follow the ReAct format described below: emit only `Thought N: <text>` then `Action N: ToolName[args]`. Never apologise, never write conversational prose in an Action line. After all data is collected with NotebookWrite, your final Action MUST be `Planner[Query]` (substituting the user's query) to produce the travel plan.

PLAN:
{plan_text}

ReAct protocol follows.

"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "agents")))

from agents.prompts import (
    planner_agent_prompt,
    cot_planner_agent_prompt,
    react_planner_agent_prompt,
    reflect_prompt,
    react_reflect_planner_agent_prompt,
    REFLECTION_HEADER,
    PromptTemplate,
)
from llm_client import LLMClient, TokenLedger
from env import ReactEnv, ReactReflectEnv
import tiktoken
import re
import time
from enum import Enum
from typing import List, Optional, Union, Literal
import argparse


OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


class ReflexionStrategy(Enum):
    """
    REFLEXION: Apply reflexion to the next reasoning trace 
    """
    REFLEXION = 'reflexion'


class Planner:
    def __init__(self,
                 agent_prompt: PromptTemplate = planner_agent_prompt,
                 model_name: str = 'gpt-4o-mini',
                 ledger: Optional[TokenLedger] = None,
                 seed: Optional[int] = None,
                 ) -> None:

        self.agent_prompt = agent_prompt
        self.scratchpad: str = ''
        self.model_name = model_name
        self.seed = seed
        try:
            self.enc = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.enc = tiktoken.get_encoding("cl100k_base")

        self.ledger = ledger if ledger is not None else TokenLedger()
        self.llm = LLMClient(
            model=model_name,
            temperature=0.0,
            max_tokens=4096,
            ledger=self.ledger,
        )

        print(f"PlannerAgent {model_name} loaded.")

    def run(self, text, query, log_file=None) -> str:
        if log_file:
            log_file.write('\n---------------Planner\n' + self._build_agent_prompt(text, query))
        prompt = self._build_agent_prompt(text, query)
        if len(self.enc.encode(prompt)) > 100000:
            return 'Max Token Length Exceeded.'
        return self.llm.chat(
            [{"role": "user", "content": prompt}],
            seed=self.seed,
            role_tag='upstream_planner_tool',
        )

    def _build_agent_prompt(self, text, query) -> str:
        return self.agent_prompt.format(
            text=text,
            query=query)


class ReactPlanner:
    """
    A question answering ReAct Agent.
    """
    def __init__(self,
                 agent_prompt: PromptTemplate = react_planner_agent_prompt,
                 model_name: str = 'gpt-4o-mini',
                 ledger: Optional[TokenLedger] = None,
                 seed: Optional[int] = None,
                 ) -> None:

        self.agent_prompt = agent_prompt
        self.ledger = ledger if ledger is not None else TokenLedger()
        self.seed = seed
        self.react_llm = LLMClient(
            model=model_name,
            temperature=0.0,
            max_tokens=1024,
            ledger=self.ledger,
        )
        self._stop = ["Action", "Thought", "Observation"]
        self.env = ReactEnv()
        self.query = None
        self.max_steps = 30
        self.reset()
        self.finished = False
        self.answer = ''
        try:
            self.enc = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.enc = tiktoken.get_encoding("cl100k_base")

    def run(self, text, query, reset = True) -> None:

        self.query = query
        self.text = text

        if reset:
            self.reset()
        

        while not (self.is_halted() or self.is_finished()):
            self.step()
        
        return self.answer, self.scratchpad

    
    def step(self) -> None:
        # Think
        self.scratchpad += f'\nThought {self.curr_step}:'
        self.scratchpad += ' ' + self.prompt_agent()
        print(self.scratchpad.split('\n')[-1])

        # Act
        self.scratchpad += f'\nAction {self.curr_step}:'
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        print(self.scratchpad.split('\n')[-1])

        # Observe
        self.scratchpad += f'\nObservation {self.curr_step}: '

        action_type, action_arg = parse_action(action)

        if action_type == 'CostEnquiry':
            try:
                input_arg = eval(action_arg)
                if type(input_arg) != dict:
                    raise ValueError('The sub plan can not be parsed into json format, please check. Only one day plan is supported.')
                observation = f'Cost: {self.env.run(input_arg)}'
            except SyntaxError:
                observation = f'The sub plan can not be parsed into json format, please check.'
            except ValueError as e:
                observation = str(e)
        
        elif action_type == 'Finish':
            self.finished = True
            observation = f'The plan is finished.'
            self.answer = action_arg
        
        else:
            observation = f'Action {action_type} is not supported.'
        
        self.curr_step += 1

        self.scratchpad += observation
        print(self.scratchpad.split('\n')[-1])

    def prompt_agent(self) -> str:
        text = self.react_llm.chat(
            [{"role": "user", "content": self._build_agent_prompt()}],
            seed=self.seed,
            stop=self._stop,
            role_tag='react_planner',
        )
        return format_step(text)

    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            query = self.query,
                            text = self.text,
                            scratchpad = self.scratchpad)

    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        return ((self.curr_step > self.max_steps) or (
                    len(self.enc.encode(self._build_agent_prompt())) > 100000)) and not self.finished

    def reset(self) -> None:
        self.scratchpad = ''
        self.answer = ''
        self.curr_step = 1
        self.finished = False


class ReactReflectPlanner:
    """
    A question answering Self-Reflecting React Agent.
    """
    def __init__(self,
                 agent_prompt: PromptTemplate = react_reflect_planner_agent_prompt,
                 reflect_prompt: PromptTemplate = reflect_prompt,
                 model_name: str = 'gpt-4o-mini',
                 ledger: Optional[TokenLedger] = None,
                 seed: Optional[int] = None,
                 ) -> None:

        self.agent_prompt = agent_prompt
        self.reflect_prompt = reflect_prompt
        self.ledger = ledger if ledger is not None else TokenLedger()
        self.seed = seed
        self.react_llm = LLMClient(model=model_name, temperature=0.0, max_tokens=1024, ledger=self.ledger)
        self.reflect_llm = LLMClient(model=model_name, temperature=0.0, max_tokens=1024, ledger=self.ledger)
        self._stop = ["Action", "Thought", "Observation,'\n"]
        self.model_name = model_name
        self.env = ReactReflectEnv()
        self.query = None
        self.max_steps = 30
        self.reset()
        self.finished = False
        self.answer = ''
        self.reflections: List[str] = []
        self.reflections_str: str = ''
        try:
            self.enc = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.enc = tiktoken.get_encoding("cl100k_base")

    def run(self, text, query, reset = True) -> None:

        self.query = query
        self.text = text

        if reset:
            self.reset()
        

        while not (self.is_halted() or self.is_finished()):
            self.step()
            if self.env.is_terminated and not self.finished:
                self.reflect(ReflexionStrategy.REFLEXION)

        
        return self.answer, self.scratchpad

    
    def step(self) -> None:
        # Think
        self.scratchpad += f'\nThought {self.curr_step}:'
        self.scratchpad += ' ' + self.prompt_agent()
        print(self.scratchpad.split('\n')[-1])

        # Act
        self.scratchpad += f'\nAction {self.curr_step}:'
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        print(self.scratchpad.split('\n')[-1])

        # Observe
        self.scratchpad += f'\nObservation {self.curr_step}: '

        action_type, action_arg = parse_action(action)

        if action_type == 'CostEnquiry':
            try:
                input_arg = eval(action_arg)
                if type(input_arg) != dict:
                    raise ValueError('The sub plan can not be parsed into json format, please check. Only one day plan is supported.')
                observation = f'Cost: {self.env.run(input_arg)}'
            except SyntaxError:
                observation = f'The sub plan can not be parsed into json format, please check.'
            except ValueError as e:
                observation = str(e)
        
        elif action_type == 'Finish':
            self.finished = True
            observation = f'The plan is finished.'
            self.answer = action_arg
        
        else:
            observation = f'Action {action_type} is not supported.'
        
        self.curr_step += 1

        self.scratchpad += observation
        print(self.scratchpad.split('\n')[-1])

    def reflect(self, strategy: ReflexionStrategy) -> None:
        print('Reflecting...')
        if strategy == ReflexionStrategy.REFLEXION: 
            self.reflections += [self.prompt_reflection()]
            self.reflections_str = format_reflections(self.reflections)
        else:
            raise NotImplementedError(f'Unknown reflection strategy: {strategy}')
        print(self.reflections_str)

    def prompt_agent(self) -> str:
        text = self.react_llm.chat(
            [{"role": "user", "content": self._build_agent_prompt()}],
            seed=self.seed,
            stop=self._stop,
            role_tag='react_reflect_planner',
        )
        return format_step(text)

    def prompt_reflection(self) -> str:
        text = self.reflect_llm.chat(
            [{"role": "user", "content": self._build_reflection_prompt()}],
            seed=self.seed,
            stop=self._stop,
            role_tag='reflect',
        )
        return format_step(text)
    
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            query = self.query,
                            text = self.text,
                            scratchpad = self.scratchpad,
                            reflections = self.reflections_str)
    
    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
                            query = self.query,
                            text = self.text,
                            scratchpad = self.scratchpad)
    
    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        return ((self.curr_step > self.max_steps) or (
                    len(self.enc.encode(self._build_agent_prompt())) > 100000)) and not self.finished

    def reset(self) -> None:
        self.scratchpad = ''
        self.answer = ''
        self.curr_step = 1
        self.finished = False
        self.reflections = []
        self.reflections_str = ''
        self.env.reset()

def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '')

def parse_action(string):
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)

    try:
        if match:
            action_type = match.group(1)
            action_arg = match.group(2)
            return action_type, action_arg
        else:
            return None, None
        
    except:
        return None, None

def format_reflections(reflections: List[str],
                        header: str = REFLECTION_HEADER) -> str:
    if reflections == []:
        return ''
    else:
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])

# if __name__ == '__main__':
    
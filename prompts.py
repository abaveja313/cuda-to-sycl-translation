import re
from typing import Any, Callable, Union, List

from langchain.agents import AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import Tool
from pydantic import BaseModel

TEMPLATE = """
Below is the block to migrate:
{block}
"""

AGENT_IT_TEMPLATE_PREFIX = """
You are a programmer whose job is to fix an error emitted from the dpc++ conversion tool, which attempts to migrate code 
from CUDA to SYCL. Where it cannot migrate automatically it leaves a comment. You should output the same code block I 
provided you with the errors corrected efficiently and comments detailing the changes you have made and the reasons why. 
You must NOT change any functionality of the original code. If no change is required, write a comment indicating that is the 
case. If there are multiple instances of an error, fix only the first one. Provide the modified code as your final answer.
Do NOT place a final answer if there are still modifications to complete!

You have access to ONLY following tools. Any other actions are disallowed:
"""
AGENT_IT_INSTRUCTIONS = """Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: One of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question. Only produce this when all modification is complete.
"""

AGENT_IT_TEMPLATE_SUFFIX = """
Instructions: {input}
{agent_scratchpad}
"""


class CustomOutputParser(AgentOutputParser):
	def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
		# Check if agent should finish
		if "Final Answer:" in llm_output:
			return AgentFinish(
				# Return values is generally always a dictionary with a single `output` key
				# It is not recommended to try anything else at the moment :)
				return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
				log=llm_output,
			)
		# Parse out the action and action input
		regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
		match = re.search(regex, llm_output, re.DOTALL)
		if not match:
			raise ValueError(f"Could not parse LLM output: `{llm_output}`")
		action = match.group(1).strip()
		action_input = match.group(2)
		# Return the action and action input
		return AgentAction(
			tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
		)


class CudaToSyclPromptTemplate(StringPromptTemplate, BaseModel):
	# TODO add validation for err codes + code + lines
	def format(self, **kwargs: Any) -> str:
		return TEMPLATE.format(**kwargs)

	def _prompt_type(self) -> str:
		return 'cuda-to-sycl'

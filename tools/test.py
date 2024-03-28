from operator import itemgetter
from typing import Union

from langchain.output_parsers import JsonOutputToolsParser
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool


@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int


@tool
def add(first_int: int, second_int: int) -> int:
    "Add two integers."
    return first_int + second_int


@tool
def exponentiate(base: int, exponent: int) -> int:
    "Exponentiate the base to the exponent power."
    return base**exponent


model = ChatOpenAI(model="gpt-3.5-turbo")
tools = [multiply, exponentiate, add]
model_with_tools = model.bind_tools(tools)
tool_map = {tool.name: tool for tool in tools}


def call_tool(tool_invocation: dict) -> Union[str, Runnable]:
    """Function for dynamically constructing the end of the chain based on the model-selected tool."""
    tool = tool_map[tool_invocation["type"]]
    return RunnablePassthrough.assign(output=itemgetter("args") | tool)


# .map() allows us to apply a function to a list of inputs.
call_tool_list = RunnableLambda(call_tool).map()
chain = model_with_tools | JsonOutputToolsParser() | call_tool_list

print(chain.invoke('what is 3x4+2*6?'))

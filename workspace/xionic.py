from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_json_agent
from langchain.agents import AgentExecutor
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import load_tools
from dotenv import load_dotenv
load_dotenv()


tools = load_tools(["serpapi"])
tools.append
tools[0].invoke({"query": "What is the weather in New York?"})


client = ChatOpenAI(
    base_url="http://sionic.chat:8001/v1",
    api_key="934c4bbc-c384-4bea-af82-1450d7f8128d",
    model="xionic-ko-llama-3-70b",
    temperature=0.1,
)

json_prompt = hub.pull("hwchase17/react-chat-json")

llama3_agent = create_json_agent(client, json_prompt)

llama3_agent_executer = AgentExecutor(
    agent=llama3_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
)

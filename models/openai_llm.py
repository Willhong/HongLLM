from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate


load_dotenv()

llm = ChatOpenAI(temperature=0,  model="gpt-4-turbo-preview")

if __name__ == "__main__":
    template = """Question: {question}

    Answer: Let's think step by step."""

    prompt = PromptTemplate.from_template(template)

    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        output_key='test',
        verbose=True,
    )

    query = "What is the name of the person who wrote the book 'The Great Gatsby'?"
    print(llm_chain.invoke({'question': query}))

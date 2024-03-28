from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains.llm import LLMChain


load_dotenv()

llm = ChatGroq(temperature=0,  model="mixtral-8x7b-32768")


if __name__ == "__main__":

    def create_chain(llm,  output_key, query):
        return LLMChain(
            llm=llm,
            prompt=ChatPromptTemplate.from_template(
                template=query
            ),
            output_key=output_key,
            verbose=True,
        )

    query = "What is the name of the person who wrote the book 'The Great Gatsby'?"
    context = {}
    print(create_chain(llm, 'test', query).invoke(context))

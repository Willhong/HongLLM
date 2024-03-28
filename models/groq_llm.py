from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains.llm import LLMChain


load_dotenv()
mixtral = "mixtral-8x7b-32768"
llama2 = "llama2-70b-4096"


def get_groq_llm(model_name):
    return ChatGroq(temperature=0,  model=model_name)


groq_mixtral = get_groq_llm(mixtral)
groq_llama2 = get_groq_llm(llama2)


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

    query = "Who is the most popular Korean in the world?"
    context = {}
    for i in create_chain(groq_mixtral, 'AI', query).stream(context):
        print(i, end="", flush=True)
    # print(create_chain(groq_llama2, 'AI', query).invoke(context))

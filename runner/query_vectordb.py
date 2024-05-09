
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
import os
import models.groq_llm as groq_llm
import models.openai_llm as openai_llm


DB_PATH = "./chroma_db/unity"

# db = Chroma.from_documents(texts, OpenAIEmbeddings(
#     disallowed_special=()), persist_directory=DB_PATH)


def query_vectordb(question):
    db = Chroma(persist_directory=DB_PATH,
                embedding_function=OpenAIEmbeddings(disallowed_special=()))
    retriever = db.as_retriever(
        search_type="mmr",  # Also test "similarity"
        search_kwargs={"k": 8},
    )

    # llm = groq_llm.groq_mixtral
    llm = openai_llm.llm
    memory = ConversationSummaryMemory(
        llm=llm, memory_key="chat_history", return_messages=True
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory)

    result = qa(question)
    result["answer"]

    print(result["answer"])

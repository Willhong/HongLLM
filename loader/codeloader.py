from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.language import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

import os

repo_path = 'C:/Users/PC/Documents/카카오톡 받은 파일/마무리_소스코드'


def save_to_db(repo_path):
    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[".cs"],
        exclude=["**/non-utf8-encoding.py"],
        parser=LanguageParser(language=Language.CSHARP, parser_threshold=500),
    )
    documents = loader.load()
    print(len(documents))

    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.CSHARP, chunk_size=2000, chunk_overlap=200
    )
    texts = python_splitter.split_documents(documents)
    print(len(texts))

    DB_PATH = "./chroma_db/unity"

    # save the documents to the database
    db = Chroma.from_documents(texts, OpenAIEmbeddings(
        disallowed_special=()), persist_directory=DB_PATH)


if __name__ == '__main__':
    save_to_db(repo_path)

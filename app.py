import runner.query_vectordb as query_vectordb

from dotenv import load_dotenv
load_dotenv()

query_vectordb.query_vectordb(
    "List all codes to add when adding a new field to a model in managesystem")

import runner.query_vectordb as query_vectordb

from dotenv import load_dotenv
load_dotenv()

query_vectordb.query_vectordb(
    "After modifying the proto file, What files should I edit to reflect the changes?")

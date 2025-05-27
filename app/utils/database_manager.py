import os
from pymongo.database import Database
from pymongo import MongoClient


db: Database = None


def init_db():
    """
    Initialize the MongoDB connection
    """
    try:
        client = MongoClient(os.getenv("MONGODB_CONNECTION_STRING"))
        globals()["db"] = client["q2forge"]
    except Exception as e:
        print("MongoDB connection failed", e)


init_db()

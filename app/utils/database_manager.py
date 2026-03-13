import os
from pymongo.database import Database
from pymongo import MongoClient


db: Database = None
client: MongoClient = None


def init_db():
    """
    Initialize the MongoDB connection
    """
    try:
        globals()["client"] = MongoClient(os.getenv("MONGODB_CONNECTION_STRING"))
        globals()["db"] = client["q2forge"]
    except Exception as e:
        print("MongoDB connection failed", e)


def close_db():
    """
    Close the MongoDB connection
    """
    if client:
        client.close()
        print("MongoDB connection closed")
    else:
        print("No MongoDB connection to close")


init_db()

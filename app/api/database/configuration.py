from app.api.responses.kg_config import KGConfig
from app.utils.database_manager import db


def get_user_active_config(username: str) -> KGConfig:

    results = db["users"].aggregate(
        [
            {"$match": {"username": username}},  # example order ID
            {
                "$lookup": {
                    "from": "configurations",
                    "localField": "active_config_id",
                    "foreignField": "_id",
                    "as": "active_config",
                }
            },
        ]
    )

    try:
        doc = results.next()["active_config"][0]
        if not doc:
            return None

        return KGConfig.from_mongo(doc)
    except Exception:
        return None


def get_configuration(kg_short_name: str) -> KGConfig:
    """
    Get a user from the database by username.

    Args:
        kg_short_name (str): The short name of the configuration to retrieve.
    Returns:
        KGConfig: The configuration object if found, None otherwise.
    Raises:
        Exception: If there is an error retrieving the configuration from the database.
    """
    doc = db["configurations"].find_one({"kg_short_name": kg_short_name})
    if not doc:
        return None

    return KGConfig.from_mongo(doc)


def get_available_configurations() -> list[str]:

    results = db["configurations"].find({}, {"kg_short_name": 1, "_id": 0})

    short_names = [result["kg_short_name"] for result in results]

    return short_names


def add_configuration(config: dict) -> KGConfig:
    """
    Add a new configuration to the database.

    Args:
        config (KGConfig): The configuration object to be added.
    Returns:
        KGConfig: The added configuration object if successful, None otherwise.
    Raises:
        Exception: If there is an error adding the configuration to the database.
    """

    insertedConfig = db["configurations"].insert_one(config)
    if not insertedConfig:
        return None

    return config

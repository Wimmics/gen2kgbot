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


def get_available_configurations() -> list[str]:

    results = db["configurations"].find({}, {"kg_short_name": 1, "_id": 0})

    short_names = [result["kg_short_name"] for result in results]

    return short_names

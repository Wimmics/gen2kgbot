from bson import ObjectId
from app.api.models.user import SparqlGenerationChat, UserInDB, UserResponse
from app.api.responses.kg_config import KGConfig
from app.utils.database_manager import db
from app.utils.logger_manager import setup_logger


logger = setup_logger(__package__, __file__)


def get_user(username: str) -> UserInDB:
    """
    Get a user from the database by username.

    Args:
        username (str): The username of the user to retrieve.
    Returns:
        UserInDB: The user object if found, None otherwise.
    Raises:
        Exception: If there is an error retrieving the user from the database.
    """
    doc = db["users"].find_one({"username": username})
    if not doc:
        return None

    return UserInDB.from_mongo(doc)


def add_user(user: UserInDB) -> UserInDB:
    """
    Add a new user to the database.

    Args:
        user (UserInDB): The user object to add.
    Returns:
        UserInDB: The added user object if successful, None otherwise.
    Raises:
        Exception: If there is an error adding the user to the database.
    """
    insertedUser = db["users"].insert_one(
        user.model_dump(by_alias=True, exclude_none=True)
    )
    if not insertedUser:
        return None

    return user


def update_active_config(user: UserResponse, kg_short_name: str) -> KGConfig:

    logger.info(f"Updating user: {user.username} active config")
    try:
        config = db["configurations"].find_one({"kg_short_name": kg_short_name})
        selected_config = KGConfig.from_mongo(config)

        if config is None:
            raise Exception("No config found matching the given short name")

        results = db["users"].update_one(
            {"username": user.username},
            [{"$set": {"active_config_id": ObjectId(selected_config.id)}}],
        )

        if results.matched_count > 0:
            updated_user = db["users"].find_one({"username": user.username})

            if (
                UserInDB.from_mongo(updated_user).active_config_id
                != user.active_config_id
            ):
                return selected_config
            else:
                logger.info(
                    f"User {user.username} already has the active config set to {kg_short_name}"
                )
                raise Exception(
                    f"User {user.username} already has the active config set to {kg_short_name}"
                )

        raise Exception("The operation did not succeed")

    except Exception as e:
        raise Exception(f"Error updating user's active config: {e}")


def update_user_chat_history(
    user: UserResponse, chat_request: SparqlGenerationChat
) -> SparqlGenerationChat:

    logger.info(f"Updating user: {user.username} SAPRQL chat history")
    try:

        user.sparql_chats.append(chat_request)

        chats = [chat.model_dump() for chat in user.sparql_chats] 

        results = db["users"].update_one(
            {"username": user.username},
            [{"$set": {"sparql_chats": chats}}],
        )

        if results.matched_count > 0:
            return chat_request

        raise Exception("The operation did not succeed")

    except Exception as e:
        raise Exception(f"Error updating user's SPARQL chat history: {e}")


def delete_user_chat_from_history(
    user: UserResponse, chat_request: SparqlGenerationChat
) -> SparqlGenerationChat:

    logger.info(f"Updating user: {user.username} SAPRQL chat history")
    try:

        chats = [chat.model_dump() for chat in user.sparql_chats if chat.id != chat_request.id]

        results = db["users"].update_one(
            {"username": user.username},
            [{"$set": {"sparql_chats": chats}}],
        )

        if results.matched_count > 0:
            return chat_request

        raise Exception("The operation did not succeed")

    except Exception as e:
        raise Exception(f"Error updating user's SPARQL chat history: {e}")

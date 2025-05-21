from app.api.models.user import UserInDB
from app.utils.config_manager import db


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
    user = db["users"].find_one({"username": username})
    if not user:
        return None

    return UserInDB(**user)


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
    insertedUser = db["users"].insert_one(user.model_dump())
    if not insertedUser:
        return None

    return user

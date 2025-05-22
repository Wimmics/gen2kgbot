import os
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from passlib.context import CryptContext
from app.api.database.user import add_user, get_user
from app.api.models.token import TokenData
from app.api.models.user import UserInDB, UserSignUp


SECRET_KEY = os.getenv("Q2FORGE_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 3600

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth_2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def verify_password(plain_password, hashed_password):
    """
    Verify if the provided password matches the hashed password.

    Args:
        plain_password (str): The plain password to verify.
        hashed_password (str): The hashed password to compare against.

    Returns:
        bool: True if the password matches, False otherwise.
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    """
    Hash a password by default using bcrypt.

    Args:
        password (str): The password to hash.

    Returns:
        str: The hashed password.
    """

    return pwd_context.hash(password)


def authenticate_user(username: str, password: str):
    """
    Authenticate a user by verifying the provided username and password.

    Args:
        username (str): The username of the user.
        password (str): The password of the user.

    Returns:
        UserInDB | bool: The authenticated user object if successful, False otherwise.
    """
    user = get_user(username)

    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False

    return user


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    """
    Create a JWT access token with an expiration time.

    Args:
        data (dict): The data to encode in the token.
        expires_delta (timedelta | None): The expiration time delta. If None, defaults to `ACCESS_TOKEN_EXPIRE_MINUTES`.

    Returns:
        str: The encoded JWT access token.
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt


async def get_current_user(token: str = Depends(oauth_2_scheme)):
    """
    Get the current user from the provided JWT token.

    Args:
        token (str): The JWT token to decode.
    Returns:
        UserInDB: The authenticated user object.
    Raises:
        HTTPException: If the token is invalid or the user does not exist.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")

        if username is None:
            raise credentials_exception

        token_data = TokenData(username=username)

    except JWTError:
        raise credentials_exception

    user = get_user(username=token_data.username)

    if user is None:
        raise credentials_exception

    return user


async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)):
    """
    Get the current active user from the provided JWT token.

    Args:
        current_user (UserInDB): The authenticated user object.
    Returns:
        UserInDB: The authenticated user object.
    Raises:
        HTTPException: If the user is inactive.
    """
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")

    return current_user


def create_new_user(new_user: UserSignUp):
    """
    Create a new user and return an access token.
    
    Args:
        new_user (UserSignUp): The user object containing the username and password.
    Returns:
        str: The access token for the newly created user.
    Raises:
        HTTPException: If the username is already registered or if the user could not be created.
    """

    if get_user(new_user.username) is not None:
        raise HTTPException(status_code=400, detail="Username already registered")

    hashed_password = get_password_hash(new_user.password)

    user = add_user(
        UserInDB(
            username=new_user.username,
            hashed_password=hashed_password,
            disabled=False,
        )
    )

    if user is None:
        raise HTTPException(status_code=400, detail="User could not be created")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    return create_access_token(
        data={"sub": new_user.username}, expires_delta=access_token_expires
    )

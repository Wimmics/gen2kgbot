from pydantic import BaseModel, Field


class UserResponse(BaseModel):
    username: str = Field(..., description="The username of the user.")
    disabled: bool | None = Field(
        default=None, description="Indicates if the user's account is disabled or not."
    )
    active_config_id: int | None = Field(
        default=None, description="The ID of the active configuration for the user."
    )


class UserInDB(UserResponse):
    hashed_password: str = Field(
        ..., description="The hashed password of the user."
    )


class UserSignUp(BaseModel):
    username: str = Field(..., description="The username of the user.")
    password: str = Field(..., description="The password of the user.")

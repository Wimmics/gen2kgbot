from pydantic import BaseModel, Field


class Token(BaseModel):
    access_token: str = Field(..., description="The access token for the user")
    token_type: str = Field(..., description="The type of token, e.g., Bearer")
    expires_in: int = Field(..., description="The number of minutes until the token expires")


class TokenData(BaseModel):
    username: str | None = Field(default=None, description="The username of the user")

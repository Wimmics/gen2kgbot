from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    sender: str = Field(..., description="The sender of the message.")
    content: str = Field(..., description="The content of the message.")
    eventType: str = Field(..., description="The type of the event, e.g., 'init'")


class SparqlGenerationChat(BaseModel):
    id: str | None = Field(
        default=None, alias="_id", description="The unique identifier of the chat."
    )
    messages: list[ChatMessage] = Field(
        default_factory=list,
        description="List of messages in the SPARQL generation chat.",
    )
    created_at: str | None = Field(
        default=None, description="The timestamp when the chat was created."
    )

    @classmethod
    def from_mongo(cls, doc: dict):
        """Convert MongoDB document to Pydantic model"""
        if "_id" in doc:
            doc["_id"] = str(doc["_id"])
        return cls(**doc)

    model_config = ConfigDict(populate_by_name=True)


class UserResponse(BaseModel):
    id: str | None = Field(
        default=None, alias="_id", description="The unique identifier of the user."
    )
    username: str = Field(..., description="The username of the user.")
    disabled: bool | None = Field(
        default=None, description="Indicates if the user's account is disabled or not."
    )
    active_config_id: str | None = Field(
        default=None, description="The ID of the active configuration for the user."
    )
    sparql_chats: list[SparqlGenerationChat] = Field(
        ..., description="List of SPARQL generation chats history of the user."
    )
    free_cq_generation_left: int = Field(
        ...,
        description="The number of free SPARQL query generations left for the user.",
    )
    free_sparql_query_answers_left: int = Field(
        ..., description="The number of free SPARQL query answers left for the user."
    )
    free_sparql_query_judging_left: int = Field(
        ..., description="The number of free SPARQL query judgings left for the user."
    )

    @classmethod
    def from_mongo(cls, doc: dict):
        """Convert MongoDB document to Pydantic model"""
        if "_id" in doc:
            doc["_id"] = str(doc["_id"])
        if "active_config_id" in doc:
            doc["active_config_id"] = str(doc["active_config_id"])
        return cls(**doc)

    model_config = ConfigDict(populate_by_name=True)


class UserInDB(UserResponse):
    hashed_password: str = Field(..., description="The hashed password of the user.")


class UserSignUp(BaseModel):
    username: str = Field(..., description="The username of the user.")
    password: str = Field(..., description="The password of the user.")

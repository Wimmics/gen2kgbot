from langchain_core.messages import AIMessageChunk


def serialize_aimessagechunk(chunk):
    """
    Serializes an AIMessageChunk object to a string.
    Args:
        chunk (AIMessageChunk): The AIMessageChunk object to serialize.
    Returns:
        str: The serialized string representation of the AIMessageChunk object.
    Raises:
        TypeError: If the provided chunk is not an instance of AIMessageChunk.
    """

    if isinstance(chunk, AIMessageChunk):
        return chunk.content
    else:
        raise TypeError(
            f"Object of type {type(chunk).__name__} is not correctly formatted for serialization"
        )

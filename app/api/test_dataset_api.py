import os
from fastapi import FastAPI
from app.api.models.test_dataset_query_request import TestDatasetQueryRequest
from app.api.services.test_dataset_service import get_query_test_answer
from fastapi.middleware.cors import CORSMiddleware


def get_env_variable(var_name: str) -> str:
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"Environement variable `{var_name}` not found.")

    return value


app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# def serialize_aimessagechunk(chunk):
#     if isinstance(chunk, AIMessageChunk):
#         return chunk.content
#     else:
#         raise TypeError(
#             f"Object of type {type(chunk).__name__} is not correctly formatted for serialization"
#         )


# async def generate_chat_events(message):
#     # async for event in scenario_6_app.astream_events(
#     #     {"initial_question": message}, version="v1"
#     # ):
#     #     if event["event"] == "on_chat_model_stream":
#     #         chunk_content = serialize_aimessagechunk(event["data"]["chunk"])
#     #         chunk_content_html = chunk_content.replace("\n", "<br>")
#     #         yield f"data: {chunk_content_html}\n\n"
#     #     elif event["event"] == "on_chat_model_end":
#     #         print("Chat model has completed its response.")

#     #     print(f"{event['metadata'].get('langgraph_node', '')} => {event['data']}")


@app.post("/api/test_dataset/judge_query")
async def test_dataset_judge_query(test_request: TestDatasetQueryRequest):
    answer = await get_query_test_answer(
        base_uri=test_request.base_uri,
        model_provider=test_request.modelProvider,
        model_name=test_request.modelName,
        question=test_request.question,
        sparql_query=test_request.sparql_query,
        sparql_query_context=test_request.sparql_query_context,
    )
    return {"result": answer}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)

import os
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from app.core.scenarios.scenario_6.scenario_6 import graph as scenario_6_app
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk


load_dotenv(find_dotenv())


def get_env_variable(var_name: str) -> str:
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"Environement variable `{var_name}` not found.")

    return value


app = FastAPI()


@app.get("/")
async def root():
    return FileResponse("static/index.html")


def serialize_aimessagechunk(chunk):
    if isinstance(chunk, AIMessageChunk):
        return chunk.content
    else:
        raise TypeError(
            f"Object of type {type(chunk).__name__} is not correctly formatted for serialization"
        )


async def generate_chat_events(message):
    async for event in scenario_6_app.astream_events(
        {"initial_question": message}, version="v1"
    ):
        if event["event"] == "on_chat_model_stream":
            chunk_content = serialize_aimessagechunk(event["data"]["chunk"])
            chunk_content_html = chunk_content.replace("\n", "<br>")
            yield f"data: {chunk_content_html}\n\n"
        elif event["event"] == "on_chat_model_end":
            print("Chat model has completed its response.")

        print(f"{event['metadata'].get('langgraph_node','')} => {event['data']}" )


@app.get("/chat_stream/{message}")
async def chat_stream_events(message: str):
    return StreamingResponse(
        generate_chat_events(message), media_type="text/event-stream"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)



from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

from langchain_rag.tf_history import chatbot_with_history

app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

add_routes(
    app,
    runnable=chatbot_with_history,
    path="/terraform",
    enabled_endpoints=["invoke", "stream", "input_schema", "output_schema", "playground"],
)

if __name__ == "__main__":
    import uvicorn

    port = 8765
    uvicorn.run(app, host="0.0.0.0", port=port)

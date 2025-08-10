# Langchain RAG

An AI-powered Retrieval-Augmented Generation (RAG) tool built with LangChain and OpenAI embeddings that ingests Terraform code files and enables users to generate their infrastructure as code in line with existing codebase.

## Usage example

Run server:
```
python langchain_rag/server.py
```

Make a request:
```
POST http://localhost:8765/terraform/invoke

{
    "input": {
        "messages": [
            {
                "type": "human",
                "content": "I need SQS queue"
            }
        ]
    },
    "config": {
        "configurable": {
            "session_id": "1"
        }
    }
}
```

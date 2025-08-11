# LangChain Terraform RAG

An end‑to‑end Retrieval‑Augmented Generation (RAG) system for Terraform
that showcases modern machine‑learning engineering practices. The
project loads existing infrastructure code, indexes it for semantic
search and uses a large language model to draft new Terraform modules
that align with the established style of your codebase.

## Key Features

- **Custom HCL splitter** – Terraform files are parsed into an abstract
  syntax tree and split by module and resource blocks, enabling
  fine‑grained retrieval and context injection.
- **LangChain & LangGraph pipeline** – Retrieval and generation steps
  are orchestrated with LangChain primitives and compiled into a
  reproducible workflow using LangGraph.
- **Chroma vector store** – Document embeddings powered by OpenAI are
  persisted in a local Chroma database for fast similarity search.
- **Redis‑backed chat history** – Conversations are stored in Redis so
  the model can produce stateful, session‑aware responses.
- **FastAPI service layer** – A lightweight API exposes the RAG workflow
  for programmatic consumption or experimentation via LangServe’s
  playground.

## Tech Stack

| Component            | Technology                                                         |
|---------------------|--------------------------------------------------------------------|
| Language model      | `gpt-4o-mini` via `langchain-openai`                                |
| Embeddings          | `text-embedding-3-small`                                           |
| Vector store        | `Chroma`                                                            |
| Orchestration       | `LangChain`, `LangGraph`, `RunnableWithMessageHistory`              |
| Parser              | `hcl2` and `lark` for HCL syntax tree generation                    |
| API server          | `FastAPI` + `langserve`                                             |
| Chat history store  | `Redis`                                                             |

## Getting Started

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables**

   ```bash
   export OPENAI_API_KEY="sk-..."
   export SOURCE_DATA_PATH="/path/to/terraform/code"
   export REDIS_URL="redis://localhost:6379/0"  # optional
   ```

3. **Run the API server**

   ```bash
   python langchain_rag/server.py
   ```

4. **Invoke the model**

   ```bash
   curl -X POST http://localhost:8765/terraform/invoke \
     -H "Content-Type: application/json" \
     -d '{
           "input": {"messages": [{"type": "human", "content": "I need an SQS queue"}]},
           "config": {"configurable": {"session_id": "1"}}
         }'
   ```

The server returns Terraform HCL that satisfies the request while
reusing variables and modules extracted from your existing codebase.

## Project Structure

```
langchain_rag/
├── server.py     # FastAPI entrypoint
├── splitter.py   # Recursive HCL block splitter
└── tf.py         # RAG assembly and graph definition
source_data/      # Sample Terraform files for indexing
```

## References

This implementation follows the Retrieval‑Augmented Generation workflow
described in the [LangChain RAG tutorial](https://python.langchain.com/docs/tutorials/rag/),
adapted for Terraform‑specific use cases.


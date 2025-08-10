import glob
import os

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_redis import RedisChatMessageHistory
from langgraph.graph import StateGraph, START, MessagesState
from typing_extensions import List, TypedDict

from langchain_rag.splitter import RecursiveHCLSplitter

load_dotenv()


# --- Load Terraform files ---
def load_terraform_files(folder_path: str):
    docs = []
    for fp in glob.glob(os.path.join(folder_path, "*.tf*")):
        with open(fp, "r", encoding="utf-8") as f:
            docs.append(Document(page_content=f.read(), metadata={"source": fp}))
    return docs


splitter = RecursiveHCLSplitter()
terraform_docs = splitter.extract_documents_from_folder("./terraform_sources")

# --- Vector store setup ---
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma(
    collection_name="terraform_docs",
    embedding_function=embedding_model,
    persist_directory="chroma_store"
)

print(f"=== {len(vector_store.get()['documents'])} DOCS IN STORE ===")
reload = False
if len(vector_store.get()["documents"]) == 0 or reload:
    ids = vector_store.get()['ids']
    if ids:
        vector_store.delete(ids=ids)

    vector_store.add_documents(terraform_docs)
    vector_store.persist()
print(f"=== {len(vector_store.get()['documents'])} DOCS IN STORE ===")

# --- Prompt and LLM ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
terraform_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a Terraform expert. Given the following extracted snippets (Context), write complete, production-ready Terraform HCL to satisfy the user’s request.

Context:
{context}

Request:
{question}

—
Output only the final Terraform code blocks, including any necessary meta-arguments (providers, version constraints, etc.).
"""
)


# --- LangGraph state ---
class State(TypedDict):
    messages: List[BaseMessage]
    context: List[Document]
    answer: str


# --- Graph steps ---
def retrieve(state: State):
    question = state["messages"][-1].content
    retrieved_docs = vector_store.similarity_search(question, k=15)

    # history.add_user_message(question)

    return {
        # "messages": history.messages,
        "messages": state["messages"],
        "context": retrieved_docs
    }


def generate(state: State):
    question = state["messages"][-1].content
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    prompt_str = terraform_prompt.format(context=docs_content, question=question)
    response = llm.invoke(prompt_str)

    # history.add_ai_message(response.content)

    return {
        # "messages": history.messages,
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "context": state["context"],
        "answer": response.content
    }


# --- Redis history store ---
REDIS_URL = os.getenv("REDIS_URL")


def get_redis_history(session_id: str) -> BaseChatMessageHistory:
    return RedisChatMessageHistory(session_id=session_id, redis_url=REDIS_URL)


# history = RedisChatMessageHistory(session_id='123456', redis_url=REDIS_URL)

graph_builder = StateGraph(state_schema=MessagesState).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")

chat_graph = graph_builder.compile()

# --- Wrap with message history ---
chatbot_with_history = RunnableWithMessageHistory(
    chat_graph,
    get_session_history=get_redis_history,
    input_messages_key="messages"
)

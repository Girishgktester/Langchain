from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

import os
from pathlib import Path
workspace_root = Path.cwd().parent
env_paths = (
    Path.cwd() / ".env",
    workspace_root / ".env",
    workspace_root / "Langchain_Basics" / ".env",
)

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path, override=True)
        print(f"Loaded environment from: {env_path}")
        break
else:
    print("No .env file found. Create Agents/.env or workspace-root/.env.")

os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "LangChainTrainings-Agents")

if os.getenv("LANGSMITH_API_KEY"):
    print(f"LangSmith tracing enabled for project: {os.environ['LANGSMITH_PROJECT']}")
else:
    print("Add LANGSMITH_API_KEY to .env to enable LangSmith tracing.")

llm = ChatOllama(
    base_url="http://localhost:11434",
    model="qwen3:8b",
    temperature=0.5,
    num_predict=2500,
)

class State(TypedDict):
    prompt: str
    response: str
    summary: str


def node_1(state: State):
    user_input = state.get("prompt", "")

    response = llm.invoke(
        f"User said: {user_input}. Respond with output, no explanation."
    )

    return {"response": response.content}


def node_2(state: State):
    response = state.get("response", "")

    summary = llm.invoke(f"Summarize this output: {response}")

    return {"summary": summary.content}


builder = StateGraph(State)

builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)

builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", END)

graph = builder.compile()

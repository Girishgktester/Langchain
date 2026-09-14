# Tool Calling in LangGraph

This notebook (`toolCalling.ipynb`) shows two ways to build a **tool-calling agent** with LangGraph. Both use a local Ollama model (`qwen3:8b`) and the same set of math tools.

## Shared setup

The first cells load environment variables (for LangSmith tracing), create the LLM, and define the tools.

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    base_url="http://localhost:11434",
    model="qwen3:8b",
    temperature=0.5,
    num_predict=2500,
)
```

```python
from langchain.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def substarct(a: int, b: int) -> int:
    """Add two numbers."""
    return a - b

@tool
def multiply(a: int, b: int) -> int:
    """Add two numbers."""
    return a * b

tools = [add, substarct, multiply]

llm_with_tools = llm.bind_tools(tools)
```

`llm.bind_tools(tools)` gives the model the option to return a **tool call** instead of a normal text answer. When the model decides a tool is needed, the response contains `tool_calls` with the tool name and its arguments.

---

## Approach 1: Without message state

This version does **not** use LangGraph's built-in `messages` state. Instead, it manages the conversation manually with a custom `State` schema.

### State

```python
class State(TypedDict):
    question: str
    tools_name: str
    tools_args: dict
    tool_result: str
    answer: str
```

Each field is a plain value, so the flow explicitly moves data from one node to the next.

### Nodes

1. **`assistant`** — sends the user question to the model. If the model asks for a tool, it stores the tool name and arguments. If not, it stores the answer directly.

```python
def assistant(state: State):
    response = llm_with_tools.invoke(
        [system_message, HumanMessage(content=state["question"])]
    )
    tool_calls = response.tool_calls
    if tool_calls:
        tool_call = tool_calls[0]
        return {
            "tools_name": tool_call["name"],
            "tools_args": tool_call["args"],
            "answer": "",
        }
    return {"tools_name": "", "tools_args": {}, "answer": response.content}
```

2. **`tools_node`** — runs the requested tool and stores the result.

```python
def tools_node(state: State):
    tool = tools_by_name[state["tools_name"]]
    result = tool.invoke(state["tools_args"])
    return {"tool_result": str(result)}
```

3. **`generate_answer`** — gives the question and the tool result back to the model to produce the final answer.

```python
def generate_answer(state: State):
    response = llm.invoke([
        system_message,
        HumanMessage(content=(
            f"Question: {state['question']}\n"
            f"The tool returned: {state['tool_result']}\n"
            "Answer the original question."
        )),
    ])
    return {"answer": response.content}
```

### Routing

The conditional edge decides whether to execute a tool or stop:

```python
def route_after_assistant(state: State):
    if state.get("tools_name"):
        return "tools_node"
    return END
```

### Graph

```python
graph = StateGraph(State)
graph.add_node("assistant", assistant)
graph.add_node("tools_node", tools_node)
graph.add_node("generate_answer", generate_answer)

graph.add_edge(START, "assistant")
graph.add_conditional_edges("assistant", route_after_assistant)
graph.add_edge("tools_node", "generate_answer")
graph.add_edge("generate_answer", END)
```

Flow: `START -> assistant -> (tools_node -> generate_answer | END)`.

---

## Approach 2: With messages state

This is the idiomatic LangGraph pattern. It uses `MessagesState`, which automatically maintains a list of chat messages.

### State

`MessagesState` is provided by LangGraph and already has a `messages` field with an `add_messages` reducer. This means every node can **append** messages instead of manually tracking each piece of data.

```python
from langgraph.graph import MessagesState
```

### Nodes

The `assistant` node returns the model's response as a message. The `messages` reducer appends it to the history.

```python
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([system_message] + state["messages"])]}
```

The tool execution is handled by LangGraph's prebuilt `ToolNode`:

```python
from langgraph.prebuilt import ToolNode

graph.add_node("tools", ToolNode(tools))
```

`ToolNode` automatically reads the pending tool calls from the last message, runs the matching tools, and appends the results as `ToolMessage`s.

### Routing

The conditional edge checks whether the model asked for a tool:

```python
def route_tools(state: MessagesState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END
```

### Graph

```python
graph = StateGraph(MessagesState)
graph.add_node("assistant", assistant)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "assistant")
graph.add_conditional_edges("assistant", route_tools)
graph.add_edge("tools", "assistant")

graph = graph.compile()
```

Flow: `START -> assistant -> tools -> assistant -> END`

After the tools run, control loops back to `assistant` so the model can use the tool results and produce a final answer. The `route_tools` conditional edge stops the loop once no more tool calls are requested.

### Invoking the graph

```python
from langchain.messages import HumanMessage

graph.invoke({"messages": [HumanMessage(content="What is 2 + 3?")]})
```

---

## Key difference

| | Without message state | With messages state |
|---|---|---|
| State | Custom fields (`question`, `tools_name`, etc.) | Built-in `messages` list |
| History | Managed manually | `add_messages` reducer |
| Tool node | Custom `tools_node` | Prebuilt `ToolNode` |
| Final answer | Separate `generate_answer` node | Loop back to `assistant` |

The **messages state** version is shorter and preferred for most agents. The **custom state** version is useful when you want fine-grained control over every piece of data in the graph.

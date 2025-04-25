from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

import os
from dotenv import load_dotenv
load_dotenv()

# Tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a (int): First number.
        b (int): Second number.
    """
    return a * b

# LLM with bount tool
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools([multiply])

# State
class State(MessagesState):
    pass

# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition,
)

builder.add_edge("tools", END)
graph = builder.compile()

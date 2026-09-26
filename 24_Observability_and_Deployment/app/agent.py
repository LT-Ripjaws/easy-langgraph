"""A small chatbot graph, exported for LangSmith Deployment.

This is the same tool-calling chatbot pattern taught in 15_Tools, packaged
as a plain Python module. The LangGraph CLI (`langgraph dev` / `langgraph up`)
imports this file and looks for the `graph` variable named in langgraph.json.
"""

from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Performs basic arithmetic operations on two numbers."""
    if operation == "add":
        result = first_num + second_num
    elif operation == "subtract":
        result = first_num - second_num
    elif operation == "multiply":
        result = first_num * second_num
    elif operation == "divide":
        if second_num == 0:
            return {"error": "Cannot divide by zero."}
        result = first_num / second_num
    else:
        return {"error": "Invalid operation. Supported operations are add, subtract, multiply, divide."}
    return {"result": result}


tools = [calculator]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState) -> dict:
    """Node that handles the chat interaction with the LLM and tools."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools)

builder = StateGraph(ChatState)
builder.add_node("chat_node", chat_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "chat_node")
builder.add_conditional_edges("chat_node", tools_condition)
builder.add_edge("tools", "chat_node")

# Module-level variable named in langgraph.json's "graphs" entry.
# No checkpointer here: the dev server and LangSmith Deployment attach their
# own persistence layer around this graph automatically.
graph = builder.compile()

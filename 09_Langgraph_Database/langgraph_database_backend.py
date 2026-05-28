from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.9)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False) # Create the database file if it doesn't exist

checkpointer = SqliteSaver(conn)

graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

bot = graph.compile(checkpointer=checkpointer)

#test

CONFIG = {'configurable': {'thread_id': '1'}}
response = bot.invoke({'messages': [HumanMessage(content="Hi i am Ripjaws, how are you?")]},
          config = CONFIG)

print(response)
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv("../.env")

llm = ChatGroq(model="llama-3.3-70b-versatile")

def chatbot_node(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

graph_builder = StateGraph(MessagesState)
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_edge(START, "chatbot")

checkpointer = InMemorySaver()
app = graph_builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "session-1"}}

response1 = app.invoke({"messages": [{"role": "user", "content": "My name is Haseeb"}]}, config)
print(response1["messages"][-1].content)

response2 = app.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}, config)
print(response2["messages"][-1].content)    
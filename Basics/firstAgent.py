print("==================My first AI Agent==================")

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.tools import tool

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

@tool
def add(a:float,b:float)->float:
    """You can add two numbers using this add tool."""
    return a+b

agent = create_agent(
    model=model,
    tools=[add]

)
query = input("Enter your question: ")

result = agent.invoke(
    {
        "messages":[
            {
                "role":"user",
                "content": query
            }
        ]
    }
)

print("\n" + "=" * 60)
print("AGENT RESULT")
print("=" * 60)

print(result)

print("\n" + "=" * 60)
print("FINAL ANSWER")
print("=" * 60)

print(result["messages"][-1].content)


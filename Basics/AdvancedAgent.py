import os
import requests
from tavily import TavilyClient

from pathlib import Path
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(
    api_key=TAVILY_API_KEY
)

API_KEY = os.getenv("OPENWEATHER_API_KEY")

@tool
def web_search(query:str)->str:
    """Get search the web for current information"""
    response= tavily_client.search(
        query=query,
        max_results=3
    )
    results = response["results"]

    return "\n\n".join(
        f"Title: {result['title']}\n"
        f"Content: {result['content']}"
        for result in results
    )
@tool
def add(a:float,b:float)->float:
    """Add two numbers."""
    return a+b

@tool
def multiply(a:float,b:float)->float:
    """Multiply two numbers"""
    return a*b

@tool
def get_weather(city: str) -> str:
    """Get the current weather information for a city."""

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric"
    }

    response = requests.get(
        url,
        params=params,
        timeout=10
    )

    data = response.json()

    if response.status_code != 200:
        return (
            f"Weather API Error: "
            f"{data.get('message', 'Unknown error')}"
        )

    temperature = data["main"]["temp"]

    humidity = data["main"]["humidity"]

    description = data["weather"][0]["description"]

    actual_city = data["name"]

    return (
        f"Weather in {actual_city}: "
        f"{description}, "
        f"temperature {temperature}°C, "
        f"humidity {humidity}%."
    )

agent = create_agent(
    model=model,
    tools=[add,multiply,get_weather,web_search]
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
print("AGENT MESSAGES")
print("=" * 60)

for message in result["messages"]:
    print("\n", message)
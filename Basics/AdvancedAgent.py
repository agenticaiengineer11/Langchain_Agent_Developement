import os
import requests

from pathlib import Path
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)


API_KEY = os.getenv("OPENWEATHER_API_KEY")

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
    tools=[add,multiply,get_weather]
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
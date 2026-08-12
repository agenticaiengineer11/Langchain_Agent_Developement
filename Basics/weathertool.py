import os
import requests

from pathlib import Path
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_groq import ChatGroq


env_path = Path(__file__).resolve().parent.parent / ".env"

load_dotenv(env_path)

API_KEY = os.getenv("OPENWEATHER_API_KEY")

print("API KEY FOUND:", bool(API_KEY))
print("API KEY LENGTH:", len(API_KEY) if API_KEY else 0)


@tool
def get_weather(city: str) -> str:
    """Get current weather information for a city."""

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

    print("\nAPI Status Code:", response.status_code)

    data = response.json()

    if response.status_code != 200:
        return (
            f"Weather API Error: "
            f"{data.get('message', 'Unknown error')}"
        )

    temperature = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    description = data["weather"][0]["description"]

    return (
        f"Weather in {city}: "
        f"{description}, "
        f"temperature {temperature}°C, "
        f"humidity {humidity}%."
    )


model = ChatGroq(
    model="llama-3.3-70b-versatile"
)


model_with_tools = model.bind_tools(
    [get_weather]
)


query = input("Enter your question: ")


response = model_with_tools.invoke(query)


print("\n" + "=" * 60)
print("INITIAL LLM RESPONSE")
print("=" * 60)

print(response)


if response.tool_calls:

    print("\n" + "=" * 60)
    print("TOOL CALL")
    print("=" * 60)

    print(response.tool_calls)


    tool_messages = []

    for tool_call in response.tool_calls:

        if tool_call["name"] == "get_weather":

            tool_result = get_weather.invoke(
                tool_call["args"]
            )

            print("\n" + "=" * 60)
            print("TOOL RESULT")
            print("=" * 60)

            print(tool_result)

            tool_message = ToolMessage(
                content=tool_result,
                tool_call_id=tool_call["id"]
            )

            tool_messages.append(tool_message)


    final_response = model_with_tools.invoke(
        [
            response,
            *tool_messages
        ]
    )

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(final_response.content)


else:

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(response.content)
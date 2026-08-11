import os
import requests

from pathlib import Path
from dotenv import load_dotenv
from langchain_core.tools import tool


# Load .env from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)


# Get API key
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

    print("Status Code:", response.status_code)

    data = response.json()

    print("API Response:", data)

    if response.status_code != 200:
        return f"Weather API Error: {data.get('message', 'Unknown error')}"

    temperature = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    description = data["weather"][0]["description"]

    return (
        f"Weather in {city}: "
        f"{description}, "
        f"temperature {temperature}°C, "
        f"humidity {humidity}%."
    )


# Test the tool
result = get_weather.invoke({
    "city": "Lahore"
})

print("\n" + "=" * 60)
print("FINAL WEATHER RESULT")
print("=" * 60)
print(result)
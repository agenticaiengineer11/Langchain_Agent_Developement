from dotenv import load_dotenv
from langchain_groq import ChatGroq

from tools_calculator import (
    add,
    subtract,
    multiply,
    divide,
    power,
    modulus
)


load_dotenv()


model = ChatGroq(
    model="llama-3.3-70b-versatile"
)


tools = [
    add,
    subtract,
    multiply,
    divide,
    power,
    modulus
]


tool_map = {
    tool.name: tool
    for tool in tools
}


model_with_tools = model.bind_tools(tools)

query = input("Enter your question: ")

response = model_with_tools.invoke(query)


print("\n" + "=" * 60)
print("TOOL CALL")
print("=" * 60)

print(response.tool_calls)


tool_results = []

for tool_call in response.tool_calls:

    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    print("\n" + "=" * 60)
    print("EXECUTING TOOL")
    print("=" * 60)

    print("Tool:", tool_name)
    print("Arguments:", tool_args)

    tool = tool_map[tool_name]

    result = tool.invoke(tool_args)

    print("Tool Result:", result)

    tool_results.append(
        {
            "tool_call_id": tool_call["id"],
            "result": result
        }
    )

print("\n" + "=" * 60)
print("TOOL RESULTS")
print("=" * 60)

for result in tool_results:
    print(result)

print("\n" + "=" * 60)
print("FINAL TOOL RESULT")
print("=" * 60)

for result in tool_results:
    print(result["result"])
from langchain_core.tools import tool


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract the second number from the first number."""
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

@tool
def divide(a: float, b: float) -> float:
    """Divide the first number by the second number."""

    if b == 0:
        raise ValueError("Cannot divide by zero.")

    return a / b


@tool
def power(a: float, b: float) -> float:
    """Raise the first number to the power of the second number."""
    return a ** b


@tool
def modulus(a: float, b: float) -> float:
    """Return the remainder after dividing the first number by the second number."""

    if b == 0:
        raise ValueError("Cannot calculate modulus with zero.")

    return a % b

calculator_tools = [
    add,
    subtract,
    multiply,
    divide,
    power,
    modulus
]

print("=" * 60)
print("AVAILABLE CALCULATOR TOOLS")
print("=" * 60)

for tool in calculator_tools:
    print(f"Name: {tool.name}")
    print(f"Description: {tool.description}")
    print(f"Schema: {tool.args_schema}")
    print("-" * 60)

    
print("\n" + "=" * 60)
print("TESTING TOOLS")
print("=" * 60)


print("Addition:", add.invoke({
    "a": 10,
    "b": 5
}))


print("Subtraction:", subtract.invoke({
    "a": 10,
    "b": 5
}))


print("Multiplication:", multiply.invoke({
    "a": 10,
    "b": 5
}))


print("Division:", divide.invoke({
    "a": 10,
    "b": 5
}))


print("Power:", power.invoke({
    "a": 2,
    "b": 5
}))


print("Modulus:", modulus.invoke({
    "a": 10,
    "b": 3
}))
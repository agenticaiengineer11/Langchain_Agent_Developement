print("===================Window Memory===================")
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

window_size = 4

store = {}

def get_session_history(session_id: int):
    if session_id not in store:
        store[session_id] = []

    return store[session_id]

def format_history(history):
    return "\n".join(
        f"{message.type}: {message.content}"
        for message in history
    )

def chat(session_id:str , question: str):
    history = get_session_history(session_id)

    recent_history = history[-window_size:]

    context = format_history(recent_history)
    prompt = f"""
You are a professional AI assistant.

Use the conversation history below to understand
the user's current question.

Conversation History:
{context}

Current Question:
{question}

Answer clearly and naturally.
"""

    response = model.invoke(prompt)

    answer = response.content

    history.append(
        HumanMessage(content=question)
    )
    history.append(
        AIMessage(content=answer)
    )

    if len(history) > window_size:
        store[session_id] = history[-window_size:]

    return answer

session_id = input("Enter session ID: ")

print("\nType 'exit' to stop.\n")


while True:

    question = input("You: ")

    if question.lower() in ["exit", "quit", "bye"]:

        print("Good Bye!")

        break

    answer = chat(
        session_id,
        question
    )

    print("\nAI:", answer)
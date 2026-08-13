from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

summary_model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

store = {}

def get_session(session_id: str):

    if session_id not in store:

        store[session_id] = {
            "history": [],
            "summary": ""
        }

    return store[session_id]

def create_summary(history, old_summary):

    conversation = "\n".join(
        f"{message.type}: {message.content}"
        for message in history
    )

    prompt = f"""
You are a conversation summarization system.

Create a concise summary of the conversation.

Existing summary:
{old_summary}

New conversation:
{conversation}

Preserve important information such as:

- User identity
- User preferences
- Goals
- Important facts
- Decisions
- Previous tasks
- Relevant context

Return only the updated summary.
"""

    response = summary_model.invoke(prompt)

    return response.content

def chat(session_id: str, question: str):

    session = get_session(session_id)

    history = session["history"]
    summary = session["summary"]

    context = ""

    if summary:
        context += f"Conversation Summary:\n{summary}\n\n"

    if history:
        context += "Recent Conversation:\n"

        for message in history:
            context += f"{message.type}: {message.content}\n"

    prompt = f"""
You are a professional AI assistant.

Use the conversation context to answer the user's question.

Conversation Context:
{context}

Current User Question:
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


    if len(history) >= 6:

        new_summary = create_summary(
            history,
            summary
        )

        session["summary"] = new_summary

        # Keep only recent messages
        session["history"] = history[-2:]

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

    
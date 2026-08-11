print("==================Entity Memory====================")

from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()


model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

entity_store = {}

def get_session_entities(session_id: str):

    if session_id not in entity_store:

        entity_store[session_id] = {}

    return entity_store[session_id]

def extract_entities(
    session_id: str,
    user_message: str
):

    entities = get_session_entities(session_id)

    prompt = f"""
You are an entity extraction system.

Extract important facts about the user from the
message below.

Existing entity information:
{entities}

User message:
{user_message}

Return the information as simple key-value pairs.

Only extract information that is explicitly present
or clearly stated by the user.

Examples:

name = Noman
programming_language = Python
framework = LangChain
goal = Agentic AI

If there are no new entities, return:
NONE
"""

    response = model.invoke(prompt)

    extracted_text = response.content.strip()

    if extracted_text == "NONE":
        return entities

    for line in extracted_text.splitlines():

        if "=" not in line:
            continue

        key, value = line.split("=", 1)

        key = key.strip()
        value = value.strip()

        if key and value:

            entities[key] = value


    return entities


def chat(
    session_id: str,
    question: str
):


    entities = extract_entities(
        session_id,
        question
    )


    entity_context = "\n".join(
        f"{key}: {value}"
        for key, value in entities.items()
    )

    prompt = f"""
You are a professional AI assistant.

Use the stored information about the user when
it is relevant to the question.

Stored User Information:
{entity_context}

Current User Question:
{question}

Answer clearly and naturally.
"""


    response = model.invoke(prompt)

    return response.content

session_id = input("Enter session ID: ")

print("\nType 'exit' to stop.\n")


while True:

    question = input("You: ")

    if question.lower() in [
        "exit",
        "quit",
        "bye"
    ]:

        print("Good Bye!")

        break


    answer = chat(
        session_id,
        question
    )


    print("\nAI:", answer)
    print("\n" + "=" * 50)

    print("ENTITY MEMORY")

    print("=" * 50)

    entities = get_session_entities(session_id)

    for key, value in entities.items():

        print(f"{key}: {value}")
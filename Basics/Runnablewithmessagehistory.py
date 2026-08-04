print("=============Runnable with message history==========")
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
model = ChatGroq(model="llama-3.3-70b-versatile")
parser= StrOutputParser()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system","You are professional Ai assistant"
        ),
        MessagesPlaceholder(variable_name="history"),
        (
            "human","{question}"
        )
    ]
)
chain = prompt | model | parser

store ={}

def get_session_history(session_id:str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]
chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)
session_id = input("Enter session ID: ")
while True:
    question = input("\nYou: ")
    if question.lower() in ["exit","quit","bye"]:
        print("Good Bye!")
        break
    response = chatbot.invoke(
        {
            "question":question
        },
        config={
            "configurable":{
                "session_id":session_id
            }
        }
    )
    print("\nAI: ", response)

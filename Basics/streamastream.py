print("===================Stream and astream use==========")
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional AI assistant"
        ),
        (
            "human",
            "{question}"
        )
    ]
)
chain = prompt | model | parser
question = input("Enter your question: ")

print("\nAI: ", end="", flush=True)
for chunk in chain.stream(
    {
        "question":question
    }
):
    print(chunk,end="",flush=True)
print()
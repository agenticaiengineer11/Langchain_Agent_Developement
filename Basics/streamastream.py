print("===================Stream and astream use==========")
from dotenv import load_dotenv
import asyncio
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
async def main():

    question = input("Enter your question: ")

    print("\nAI: ", end="", flush=True)

    async for chunk in chain.astream(
        {
            "question": question
        }
    ):
        print(chunk, end="", flush=True)

    print()

asyncio.run(main())
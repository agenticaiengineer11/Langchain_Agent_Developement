print("================ Runnable Retry ================")

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional Python teacher."
        ),
        (
            "human",
            "{question}"
        )
    ]
)

chain = prompt | model | parser


retry_chain = chain.with_retry(
    stop_after_attempt=3
)

question = input("Enter your question: ")

response = retry_chain.invoke(
    {
        "question": question
    }
)

print("\n" + "=" * 60)
print("MODEL RESPONSE")
print("=" * 60)
print(response)
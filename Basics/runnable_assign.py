from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")
parser = StrOutputParser()
summary_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a professional Python teacher."),
        ("human", "Give a short summary of {topic}.")
    ]
)

summary_chain = summary_prompt | model | parser

code_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert Python developer."),
        ("human", "Give a Python code example for {topic}.")
    ]
)

code_chain = code_prompt | model | parser

interview_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a Python interviewer."),
        ("human", "Generate 5 Python interview questions about {topic}.")
    ]
)

interview_chain = interview_prompt | model | parser
chain = RunnablePassthrough().assign(
    summary=summary_chain,
    code=code_chain,
    interview_questions=interview_chain
)

topic = input("Enter Topic: ")

response = chain.invoke(
    {
        "topic": topic
    }
)

print("\nOriginal Topic:")
print(response["topic"])

print("\nSummary:")
print(response["summary"])

print("\nCode:")
print(response["code"])

print("\nInterview Questions:")
print(response["interview_questions"])
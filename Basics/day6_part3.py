from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
model = ChatGroq(model = "llama-3.3-70b-versatile")

parser = StrOutputParser()
summary_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are professional python teacher."),
        ("human", "give a short summary of {topic}.")
    ]
)
code_prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are an expert python developer."),
        ("human", "Give the python code example for {topic}.")
    ]
)
interviewer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are python Interviewer."),
        ("human", "Generate 5 interview question about {topic}")
    ]
)
summary_chain = summary_prompt | model | parser
code_chain = code_prompt | model | parser
interview_chain = interviewer_prompt | model | parser
parallel_chain= RunnableParallel(
    summary = summary_chain,
    code = code_chain,
    Interview_question = interview_chain
)
topic = input("Enter your topic: ")
response = parallel_chain.invoke(
    {
         "topic":topic
    }
   
)
print("="*60)
print("Summary")
print("="*60)
print(response["summary"])
print("="*60)
print("Code")
print("="*60)
print(response["code"])
print("="*60)
print("Interview Question")
print("="*60)
print(response["Interview_question"])
import pandas as pd

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent


load_dotenv()

df = pd.read_csv("students.csv")

model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

agent = create_pandas_dataframe_agent(
    llm=model,
    df=df,
    verbose=True,
    allow_dangerous_code=True
)
query = input("Ask a question about the CSV: ")


result = agent.invoke(query)

print("\n" + "=" * 60)
print("FINAL ANSWER")
print("=" * 60)

print(result["output"])
print("====================SQL AGENT===================================")
import sqlite3
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")



db = SQLDatabase.from_uri(
    "sqlite:///students.db"
)
agent = create_sql_agent(
    llm=model,
    db=db,
    verbose=True
)

connection= sqlite3.connect("students.db")

cursor = connection.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER,
    department TEXT,
    marks INTEGER
)
""")
students = [
    (1, "Ali", 21, "CS", 85),
    (2, "Ahmed", 22, "IT", 78),
    (3, "Noman", 22, "SE", 91),
    (4, "Hassan", 21, "CS", 88),
    (5, "Usman", 23, "IT", 74)
]

cursor.executemany(
    """
    INSERT OR REPLACE INTO students
    (id, name, age, department, marks)
    VALUES (?, ?, ?, ?, ?)
    """,
    students
)

query = input("Ask a question about students: ")

result = agent.invoke(query)

print("\n" + "=" * 60)
print("FINAL ANSWER")
print("=" * 60)

print(result["output"])

connection.commit()

connection.close()
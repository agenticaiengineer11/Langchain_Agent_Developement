print("====================SQL AGENT===================================")
import sqlite3
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri(
    "sqlite:///students.db"
)
print(db.get_usable_table_names())

print(
    db.run("SELECT * FROM students")
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

connection.commit()

connection.close()
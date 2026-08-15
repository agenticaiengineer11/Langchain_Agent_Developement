import sqlite3

from langchain_community.utilities import SQLDatabase


connection = sqlite3.connect("students.db")

cursor = connection.cursor()

cursor.execute("SELECT * FROM students")

rows = cursor.fetchall()

for row in rows:
    print(row)

connection.close()

db = SQLDatabase.from_uri(
    "sqlite:///students.db"
)


print("\nTables:")
print(db.get_usable_table_names())


print("\nDatabase Result:")

print(
    db.run("SELECT * FROM students")
)
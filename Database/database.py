import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

class Database():
    def __init__(self, database_name, table_name, user, password, host):
        self.table_name = table_name
        self.connection = psycopg2.connect(
            user=user,
            dbname=database_name,
            password=password,
            host=host
        )
        self.connection.autocommit = True
        self.cursor = self.connection.cursor()
        register_vector(self.connection)

    def insert(self, data):
        execute_values(self.cursor, f"INSERT INTO {self.table_name} (name, embedding, song_fragment) VALUES %s", data)

    def similarity_search(self, embedding):
        self.cursor.execute(f"SELECT name, song_fragment FROM {self.table_name} ORDER BY embedding <-> %s LIMIT 1;", (embedding,))
        print(self.cursor.fetchall())

    def disconnect(self):
        self.cursor.close()
        self.connection.close()

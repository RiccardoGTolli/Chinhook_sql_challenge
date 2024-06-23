import psycopg2
from psycopg2 import OperationalError
import pandas as pd

def create_connection(db_name, db_user, db_password, db_host, db_port):
    connection = None
    try:
        connection = psycopg2.connect(
            database=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
        )
        print("Connection to PostgreSQL DB successful")
    except OperationalError as e:
        print(f"The error '{e}' occurred")
    return connection

def execute_read_query(connection, query):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        df = pd.DataFrame(result, columns=[desc[0] for desc in cursor.description])
        return df
    except OperationalError as e:
        print(f"The error '{e}' occurred")
    finally:
        cursor.close()
        
    

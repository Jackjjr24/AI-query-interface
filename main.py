import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
import speech_recognition as sr
import pyttsx3
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import threading
from sqlalchemy import create_engine, inspect, text, MetaData
import psycopg2
from psycopg2 import sql
import pyrebase
from graphviz import Digraph
import subprocess
import shutil


load_dotenv()


groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    st.error("GROQ_API_KEY is not set in the environment variables. Please check your .env file.")
    st.stop()


firebase_config = {
    "apiKey": "AIzaSyCjzREVSettxKtv12WsqQBToCY4_Sb1d38",
    "authDomain": "aiquery-ca10a.firebaseapp.com",
    "projectId": "aiquery-ca10a",
    "storageBucket": "aiquery-ca10a.appspot.com",
    "messagingSenderId": "394116870559",
    "appId": "1:394116870559:web:8eb59e0cceeec839714daa",
    "measurementId": "G-H8XBH1ERQD",
    "databaseURL": "https://aiquery-ca10a.firebaseio.com"
}


firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()


tts_engine = pyttsx3.init()

def speak_text(text):
    try:
        tts_engine.setProperty('rate', 150)  
        tts_engine.setProperty('volume', 1)  
        tts_engine.say(text)
        tts_engine.runAndWait()
        tts_engine.endLoop()  
    except RuntimeError as e:
        if str(e) == "run loop already started":
            tts_engine.endLoop()
            tts_engine.say(text)
            tts_engine.runAndWait()
        else:
            st.error(f"Error in TTS: {e}")
    
    tts_thread = threading.Thread(target=lambda: speak_text(text))
    tts_thread.start()

def init_database(db_type: str, user: str, password: str, host: str, port: str, database: str):
    if db_type == "MySQL":
        db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    elif db_type == "PostgreSQL":
        db_uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    else:
        raise ValueError("Unsupported database type")
    return SQLDatabase.from_uri(db_uri), db_uri

def get_sql_chain(db, db_type):
    template = """
    You are a data analyst at an organization. You are interacting with a user who is asking you questions about the organization's database.
    Based on the table schema below, write a SQL query and natural language response that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: Who lives in New York?
    SQL Query: SELECT name FROM clients WHERE City = 'New York';
    Question: What are the states in the database?
    SQL Query: SELECT state FROM clients;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """

    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0, groq_api_key=groq_api_key)
    
    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query: str, db: SQLDatabase, chat_history: list, db_type: str):
    sql_chain = get_sql_chain(db, db_type)
    
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    
    <SCHEMA>{schema}</SCHEMA>
    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0, groq_api_key=groq_api_key)
    
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })

def create_database(db_type: str, user: str, password: str, host: str, port: str, database_name: str):
    try:
        if db_type == "MySQL":
            import mysql.connector
            conn = mysql.connector.connect(user=user, password=password, host=host, port=port)
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE {database_name}")
            conn.commit()
            cursor.close()
            conn.close()
        elif db_type == "PostgreSQL":
            conn = psycopg2.connect(dbname='postgres', user=user, password=password, host=host, port=port)
            conn.autocommit = True
            cursor = conn.cursor()
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name)))
            cursor.close()
            conn.close()
        else:
            raise ValueError("Unsupported database type")
        
        st.success(f"Database '{database_name}' created successfully!")
    except Exception as e:
        st.error(f"Error creating database: {e}")

def generate_er_diagram(db_uri: str, output_file: str = "er_diagram"):
    try:
        engine = create_engine(db_uri, pool_pre_ping=True)
        inspector = inspect(engine)
        
        dot = Digraph(format='png')

        for table_name in inspector.get_table_names():
            dot.node(table_name, table_name)
            
            for column in inspector.get_columns(table_name):
                dot.node(f"{table_name}_{column['name']}", column['name'], shape='ellipse')
                dot.edge(table_name, f"{table_name}_{column['name']}")
                
        for table_name in inspector.get_table_names():
            foreign_keys = inspector.get_foreign_keys(table_name)
            for fk in foreign_keys:
                for column_name in fk['constrained_columns']:
                    dot.edge(table_name, fk['referred_table'], label=column_name)
        
        file_path = dot.render(filename=output_file, format='png', cleanup=False)
        if os.path.exists(file_path):
            return file_path
        else:
            st.error(f"File '{file_path}' not found.")
            return None
    except Exception as e:
        st.error(f"Error generating ER diagram: {e}")
        return None

def save_log(email, prompt, response):
    log_dir = "logs"
    log_file_path = os.path.join(log_dir, "query_logs.csv")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = pd.DataFrame([{
        "Email": email,
        "DateTime": timestamp,
        "Prompt": prompt,
        "Response": response
    }])
    
    os.makedirs(log_dir, exist_ok=True)
    
    if os.path.exists(log_file_path):
        log_entry.to_csv(log_file_path, mode='a', header=False, index=False)
    else:
        log_entry.to_csv(log_file_path, mode='w', header=True, index=False)

def export_logs():
    log_file_path = "logs/query_logs.csv"
    if os.path.exists(log_file_path):
        with open(log_file_path, "rb") as f:
            st.download_button(label="Download Logs", data=f, file_name="query_logs.csv")
    else:
        st.warning("No logs found.")

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I did not understand that."
    except sr.RequestError:
        return "Sorry, there was an error with the speech recognition service."

def execute_sql_file(connection, sql_file):
    cursor = connection.cursor()
    with open(sql_file, 'r') as file:
        sql_content = file.read()
    
    statements = sql_content.split(';')
    
    try:
        for statement in statements:
            if statement.strip():
                cursor.execute(statement)
        connection.commit()
        st.success("SQL file executed successfully!")
    except Exception as e:
        connection.rollback()
        st.error(f"Error executing SQL file: {e}")
    finally:
        cursor.close()

def create_snapshot(db_uri: str, snapshot_name: str, user: str, password: str, host: str, port: str):
    try:
        mysqldump_path = "C:\\Program Files\\MySQL\\MySQL Server 8.0\\bin\\mysqldump.exe"
        
        if not os.path.isfile(mysqldump_path):
            st.error(f"mysqldump executable not found at {mysqldump_path}.")
            return
        
        database_name = db_uri.split('/')[-1]
        
        dump_file = f"{snapshot_name}.sql"
        
        result = subprocess.run(
            [mysqldump_path, "-u", user, f"-p{password}", "-h", host, "-P", port, database_name],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=False
        )
        
        with open(dump_file, 'w', encoding='utf-8') as file:
            file.write(result.stdout)
        
        if result.returncode == 0:
            st.success(f"Snapshot '{snapshot_name}' created successfully!")
        else:
            st.error(f"Error creating snapshot: {result.stderr}")
            st.write("stdout:", result.stdout)
            st.write("stderr:", result.stderr)
    
    except Exception as e:
        st.error(f"Unexpected error: {e}")



def list_snapshots():
    snapshot_files = [f for f in os.listdir('.') if f.endswith('.sql')]
    return snapshot_files

def restore_snapshot(db_uri: str, snapshot_name: str, user: str, password: str, host: str, port: str):
    try:
        dump_file = f"{snapshot_name}"

        if not os.path.isfile(dump_file):
            st.error(f"Snapshot file '{dump_file}' does not exist.")
            return

        mysql_path = "C:\\Program Files\\MySQL\\MySQL Server 8.0\\bin\\mysql.exe"
        
         
        if not os.path.isfile(mysql_path):
            st.error(f"mysql executable not found at {mysql_path}.")
            return
        
         
        database_name = db_uri.split('/')[-1]
        
        with open(dump_file, 'r', encoding='utf-8') as file:
            sql_content = file.read()
        
        result = subprocess.run(
            [mysql_path, "-u", user, f"-p{password}", "-h", host, "-P", port, database_name],
            input=sql_content, text=True, shell=False, capture_output=True
        )
        
        if result.returncode == 0:
            st.success(f"Snapshot '{snapshot_name}' restored successfully!")
        else:
            st.error(f"Error restoring snapshot: {result.stderr}")
            st.write("stdout:", result.stdout)
            st.write("stderr:", result.stderr)
    
    except Exception as e:
        st.error(f"Unexpected error: {e}")



def main():
    st.sidebar.title("User Authentication")

    if 'email' not in st.session_state:
        st.session_state.email = ""
    
    if not st.session_state.email:
        action = st.sidebar.radio("Action", ["Login", "Register"])
        email = st.sidebar.text_input("Email", "")
        password = st.sidebar.text_input("Password", "", type="password")

        if action == "Login":
            if st.sidebar.button("Login"):
                try:
                    user = auth.sign_in_with_email_and_password(email, password)
                    st.sidebar.success("Logged in successfully!")
                    st.session_state.email = email
                except Exception as e:
                    st.sidebar.error(f"Login failed: {e}")
        elif action == "Register":
            if st.sidebar.button("Register"):
                try:
                    auth.create_user_with_email_and_password(email, password)
                    st.sidebar.success("User registered successfully!")
                except Exception as e:
                    st.sidebar.error(f"Registration failed: {e}")
    else:
        st.sidebar.write(f"Logged in as {st.session_state.email}")
        if st.sidebar.button("Logout"):
            st.session_state.email = ""

        st.sidebar.title("Database Configuration")

        db_type = st.sidebar.selectbox("Select Database Type", ["MySQL", "PostgreSQL"])

        default_mysql = {
            "host": "127.0.0.1",
            "port": "3306",
            "user": "root",
            "password": "",
            "database": "client_data_db"
        }
        
        default_postgres = {
            "host": "127.0.0.1",
            "port": "5432",
            "user": "postgres",
            "password": "",
            "database": "client_data_db"
        }

        host, port, user, password, database = "", "", "", "", ""

        if db_type == "MySQL":
            host = st.sidebar.text_input("Host", default_mysql["host"])
            port = st.sidebar.text_input("Port", default_mysql["port"])
            user = st.sidebar.text_input("Username", default_mysql["user"])
            password = st.sidebar.text_input("Database Password", default_mysql["password"], type="password")
            database = st.sidebar.text_input("Database", default_mysql["database"])
        elif db_type == "PostgreSQL":
            host = st.sidebar.text_input("Host", default_postgres["host"])
            port = st.sidebar.text_input("Port", default_postgres["port"])
            user = st.sidebar.text_input("Username", default_postgres["user"])
            password = st.sidebar.text_input("Database Password", default_postgres["password"], type="password")
            database = st.sidebar.text_input("Database", default_postgres["database"])

        db_uri = f"{db_type.lower()}+mysqlconnector://{user}:{password}@{host}:{port}/{database}" if db_type == "MySQL" else f"{db_type.lower()}+psycopg2://{user}:{password}@{host}:{port}/{database}"

        if st.sidebar.button("Connect to Database"):
            try:
                st.session_state.db, _ = init_database(db_type, user, password, host, port, database)
                st.session_state.db_uri = db_uri
                st.success(f"Connected to {db_type} database successfully!")
            except Exception as e:
                st.error(f"Error connecting to database: {e}")

        st.sidebar.title("Create Database")
        with st.sidebar.expander("Create Database"):
            database_name = st.text_input("New Database Name", "")
            if st.button("Create Database"):
                if database_name:
                    create_database(db_type, user, password, host, port, database_name)
                else:
                    st.warning("Please enter a database name.")

    st.sidebar.title("Load Database")
    with st.sidebar.expander("Load Database"):
        uploaded_file = st.file_uploader("Choose a .sql file", type=["sql"])
        if st.button("Load Database"):
            if uploaded_file is not None:
                if 'db' in st.session_state and st.session_state.db:
                    with open("temp.sql", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    try:
                        if db_type == "MySQL":
                            import mysql.connector
                            connection = mysql.connector.connect(user=user, password=password, host=host, port=port, database=database)
                            execute_sql_file(connection, "temp.sql")
                            connection.close()
                        elif db_type == "PostgreSQL":
                            connection = psycopg2.connect(user=user, password=password, host=host, port=port, database=database)
                            execute_sql_file(connection, "temp.sql")
                            connection.close()
                        else:
                            st.error("Unsupported database type")
                        
                        st.session_state.db, _ = init_database(db_type, user, password, host, port, database)
                    except Exception as e:
                        st.error(f"Error loading database: {e}")
                    finally:
                        os.remove("temp.sql")
                else:
                    st.error("Database not connected.")
            else:
                st.warning("Please upload a .sql file.")

    st.sidebar.title("Database Snapshots")
    snapshot_action = st.sidebar.selectbox("Action", ["Create Snapshot", "Restore Snapshot"])

    user = st.sidebar.text_input("MySQL Username", "")
    password = st.sidebar.text_input("MySQL Password", "", type="password")
    host = st.sidebar.text_input("MySQL Host", "127.0.0.1")
    port = st.sidebar.text_input("MySQL Port", "3306")

    if snapshot_action == "Create Snapshot":
        snapshot_name = st.sidebar.text_input("Snapshot Name", "")
        if st.sidebar.button("Create"):
            if snapshot_name:
                create_snapshot(st.session_state.db_uri, snapshot_name, user, password, host, port)
            else:
                st.sidebar.error("Please enter a snapshot name.")

    elif snapshot_action == "Restore Snapshot":
        snapshots = list_snapshots()
        if snapshots:
            snapshot_to_restore = st.sidebar.selectbox("Select Snapshot", snapshots)
            if st.sidebar.button("Restore"):
                restore_snapshot(st.session_state.db_uri, snapshot_to_restore, user, password, host, port)
        else:
            st.sidebar.write("No snapshots available.")
            
    st.title("AI-Powered Data Query Interface")

    tab1, tab2, tab3 = st.tabs(["Query Interface", "ER Diagram", "Logs"])

    with tab1:
        enable_tts = st.checkbox("Enable voice output", value=False)

        query = st.text_input("Enter your query:")
        submit_button = st.button("Submit")

        if submit_button and query:
            if 'db' in st.session_state and st.session_state.db:
                response = get_response(query, st.session_state.db, [], db_type)
                st.write("Response:", response)
                if enable_tts:
                    speak_text(response)
                save_log(st.session_state.email, query, response)
            else:
                st.error("Database not connected.")

        if st.button("Speak"):
            if 'db' in st.session_state and st.session_state.db:
                query = recognize_speech()
                st.text_input("Voice Query:", value=query, key="voice_query")
                if query:
                    response = get_response(query, st.session_state.db, [], db_type)
                    st.write("Response:", response)
                    if enable_tts:
                        speak_text(response)
                    save_log(st.session_state.email, query, response)
            else:
                st.error("Database not connected.")

    with tab2:
        if st.button("Generate ER Diagram"):
            if 'db' in st.session_state and st.session_state.db:
                output_file = generate_er_diagram(st.session_state.db_uri)
                if output_file:
                    st.image(output_file)
            else:
                st.error("Database not connected.")

    with tab3:
        export_logs()

if __name__ == "__main__":
    main()

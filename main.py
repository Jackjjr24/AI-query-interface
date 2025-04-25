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
from sqlalchemy import create_engine, inspect, MetaData, text
import psycopg2
from psycopg2 import sql
import pyrebase
from graphviz import Digraph
import subprocess
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    st.error("GROQ_API_KEY is not set in the environment variables. Please check your .env file.")
    st.stop()

firebase_config = {
    "apiKey": "AIzaSyCSA_R9roVKQoutfckEtICIl-D_LTBoQJk",
    "authDomain": "aiquery-19fd4.firebaseapp.com",
    "projectId": "aiquery-19fd4",
    "databaseURL": "https://aiquery-19fd4.firebaseio.com",
    "storageBucket": "aiquery-19fd4.appspot.com",
    "messagingSenderId": "587632504080",
    "appId": "1:587632504080:web:a2ee4b01d71d7947f44995",
    "measurementId": "G-L4NSQCFB8H"
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
    
    llm = ChatGroq(model="mistral-saba-24b", temperature=0, groq_api_key=groq_api_key)
    
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
    llm = ChatGroq(model="mistral-saba-24b", temperature=0, groq_api_key=groq_api_key)
    
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

def save_chat_history(email, prompt, response):
    """Save chat history to a CSV file"""
    history_dir = "chat_history"
    history_file_path = os.path.join(history_dir, f"{email}_chat_history.csv")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    history_entry = pd.DataFrame([{
        "Email": email,
        "DateTime": timestamp,
        "Prompt": prompt,
        "Response": response
    }])
    
    os.makedirs(history_dir, exist_ok=True)
    
    if os.path.exists(history_file_path):
        history_entry.to_csv(history_file_path, mode='a', header=False, index=False)
    else:
        history_entry.to_csv(history_file_path, mode='w', header=True, index=False)

def load_chat_history(email):
    """Load chat history for a specific user"""
    history_dir = "chat_history"
    history_file_path = os.path.join(history_dir, f"{email}_chat_history.csv")
    
    if os.path.exists(history_file_path):
        return pd.read_csv(history_file_path)
    return pd.DataFrame(columns=["Email", "DateTime", "Prompt", "Response"])

def export_chat_history(email):
    """Export chat history for the current user"""
    history_file_path = os.path.join("chat_history", f"{email}_chat_history.csv")
    if os.path.exists(history_file_path):
        with open(history_file_path, "rb") as f:
            # Generate unique key using email and timestamp
            unique_key = f"download_{email}_{datetime.now().timestamp()}"
            st.download_button(
                label="Download Chat History",
                data=f,
                file_name=f"{email}_chat_history.csv",
                key=unique_key
            )
    else:
        st.warning("No chat history found.")

def clear_chat_history(email):
    """Clear chat history for the current user"""
    history_file_path = os.path.join("chat_history", f"{email}_chat_history.csv")
    if os.path.exists(history_file_path):
        os.remove(history_file_path)
        st.success("Chat history cleared successfully!")
    else:
        st.warning("No chat history to clear.")

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
    with open(sql_file, 'r', encoding='utf-8') as file:
        sql_content = file.read()
    
    # Split statements and modify problematic statements
    statements = []
    for statement in sql_content.split(';'):
        statement = statement.strip()
        if not statement:
            continue
        
        # Modify CREATE TABLE statements to include IF NOT EXISTS
        if 'CREATE TABLE' in statement.upper() and 'IF NOT EXISTS' not in statement.upper():
            statement = statement.replace('CREATE TABLE', 'CREATE TABLE IF NOT EXISTS')
        
        # Modify INSERT statements to handle potential data issues
        if 'INSERT INTO' in statement.upper():
            statement = statement.replace('INSERT INTO', 'INSERT IGNORE INTO')
        
        statements.append(statement)
    
    success_count = 0
    error_count = 0
    for statement in statements:
        if statement.strip():
            try:
                cursor.execute(statement)
                success_count += 1
            except Exception as e:
                error_count += 1
                st.warning(f"Warning executing statement: {e}\nStatement: {statement[:100]}...")
                connection.rollback()  # Rollback only the failed statement
    
    connection.commit()
    
    if error_count == 0:
        st.success(f"SQL file '{sql_file}' executed successfully with {success_count} statements!")
    else:
        st.warning(f"SQL file '{sql_file}' executed with {success_count} successful statements and {error_count} errors")
    
    cursor.close()
    return error_count == 0

def create_snapshot(db_type: str, user: str, password: str, host: str, port: str, database_name: str, snapshot_path: str):
    try:
        if db_type == "MySQL":
            mysqldump_path = shutil.which('mysqldump')
            if not mysqldump_path:
                st.error("mysqldump not found in system PATH. Please ensure MySQL client tools are installed.")
                return

            command = [
                mysqldump_path,
                f"--user={user}",
                f"--password={password}",
                f"--host={host}",
                f"--port={port}",
                database_name,
                f"--result-file={snapshot_path}"
            ]
            
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                st.success(f"MySQL snapshot created successfully at {snapshot_path}!")
            else:
                st.error(f"Error creating MySQL snapshot: {result.stderr}")
                
        elif db_type == "PostgreSQL":
            pg_dump_path = shutil.which('pg_dump')
            if not pg_dump_path:
                st.error("pg_dump not found in system PATH. Please ensure PostgreSQL client tools are installed.")
                return

            command = [
                pg_dump_path,
                f"--host={host}",
                f"--port={port}",
                f"--username={user}",
                f"--dbname={database_name}",
                f"--file={snapshot_path}"
            ]
            
            env = os.environ.copy()
            env['PGPASSWORD'] = password
            
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            if result.returncode == 0:
                st.success(f"PostgreSQL snapshot created successfully at {snapshot_path}!")
            else:
                st.error(f"Error creating PostgreSQL snapshot: {result.stderr}")
        else:
            st.error("Unsupported database type for snapshot operation.")
    except Exception as e:
        st.error(f"Error creating snapshot: {e}")

def list_snapshots():
    snapshot_files = [f for f in os.listdir('.') if f.endswith('.sql')]
    return snapshot_files

def restore_snapshot(db_uri: str, snapshot_name: str, user: str, password: str, host: str, port: str):
    try:
        if not os.path.isfile(snapshot_name):
            st.error(f"Snapshot file '{snapshot_name}' does not exist.")
            return

        if 'mysql' in db_uri.lower():
            mysql_path = shutil.which('mysql')
            if not mysql_path:
                st.error("MySQL client not found in system PATH. Please ensure MySQL client tools are installed.")
                return

            database_name = db_uri.split('/')[-1]
            
            cmd = [
                mysql_path,
                f"--user={user}",
                f"--password={password}",
                f"--host={host}",
                f"--port={port}",
                database_name
            ]
            
            with open(snapshot_name, 'r', encoding='utf-8') as f:
                result = subprocess.run(cmd, stdin=f, capture_output=True, text=True)
                
        elif 'postgresql' in db_uri.lower():
            psql_path = shutil.which('psql')
            if not psql_path:
                st.error("PostgreSQL client not found in system PATH. Please ensure PostgreSQL client tools are installed.")
                return

            database_name = db_uri.split('/')[-1].split('?')[0]
            
            cmd = [
                psql_path,
                f"--host={host}",
                f"--port={port}",
                f"--username={user}",
                f"--dbname={database_name}",
                f"--file={snapshot_name}"
            ]
            
            env = os.environ.copy()
            env['PGPASSWORD'] = password
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        else:
            st.error("Unsupported database type for restoration")
            return

        if result.returncode == 0:
            st.success(f"Snapshot '{snapshot_name}' restored successfully!")
            db_type = "MySQL" if 'mysql' in db_uri.lower() else "PostgreSQL"
            st.session_state.db, st.session_state.db_uri = init_database(
                db_type, user, password, host, port, database_name
            )
        else:
            st.error(f"Error restoring snapshot: {result.stderr}")
            st.text(f"Command executed: {' '.join(cmd)}")
            
    except Exception as e:
        st.error(f"Unexpected error during snapshot restoration: {str(e)}")

def get_dataframe_from_query(db_uri: str, query: str):
    """Execute SQL query and return results as pandas DataFrame"""
    try:
        engine = create_engine(db_uri)
        with engine.connect() as connection:
            result = connection.execute(text(query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        return df
    except Exception as e:
        st.error(f"Error executing query: {e}")
        return None

def get_table_data(db_uri: str, table_name: str, limit=1000):
    """Get data from a specific table"""
    return get_dataframe_from_query(db_uri, f"SELECT * FROM {table_name} LIMIT {limit}")

def get_table_info(db_uri: str):
    """Get information about tables and columns in the database"""
    try:
        engine = create_engine(db_uri)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        table_info = {}
        for table in tables:
            columns = inspector.get_columns(table)
            table_info[table] = {
                'columns': [col['name'] for col in columns],
                'primary_key': inspector.get_pk_constraint(table)['constrained_columns'],
                'foreign_keys': inspector.get_foreign_keys(table)
            }
        return table_info
    except Exception as e:
        st.error(f"Error getting table info: {e}")
        return None

def visualize_data_interactive(df):
    """Create interactive visualizations based on dataframe content"""
    st.write("### Interactive Data Visualization")
    
    if df is None or df.empty:
        st.warning("No data available for visualization")
        return
    
    # Show raw data
    with st.expander("View Raw Data"):
        st.dataframe(df)
    
    # Basic statistics
    with st.expander("Basic Statistics"):
        st.write(df.describe(include='all'))
    
    # Visualization options
    st.write("### Visualization Options")
    
    # Determine column types
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    
    # Visualization type selection
    viz_type = st.selectbox(
        "Select visualization type",
        ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", 
         "Pie Chart", "Heatmap", "Violin Plot", "Pair Plot"]
    )
    
    try:
        if viz_type == "Bar Chart":
            if not categorical_cols:
                st.warning("No categorical columns found for bar chart")
                return
            
            x_axis = st.selectbox("Select category column", categorical_cols)
            y_axis = st.selectbox("Select value column", numeric_cols)
            
            if st.checkbox("Show as horizontal bar chart"):
                fig = px.bar(df, y=x_axis, x=y_axis, title=f"{y_axis} by {x_axis}")
            else:
                fig = px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
            
            if len(categorical_cols) > 1:
                color_col = st.selectbox("Select color column", [None] + categorical_cols)
                if color_col:
                    fig.update_traces(marker_coloraxis=None)
                    fig = px.bar(df, x=x_axis, y=y_axis, color=color_col, title=f"{y_axis} by {x_axis} colored by {color_col}")
            
            st.plotly_chart(fig)
        
        elif viz_type == "Line Chart":
            if not numeric_cols:
                st.warning("No numeric columns found for line chart")
                return
            
            x_axis = st.selectbox("Select X-axis column", df.columns)
            y_axis = st.selectbox("Select Y-axis column", numeric_cols)
            
            fig = px.line(df, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}")
            
            if len(numeric_cols) > 1:
                color_col = st.selectbox("Select color column", [None] + categorical_cols)
                if color_col:
                    fig = px.line(df, x=x_axis, y=y_axis, color=color_col, title=f"{y_axis} over {x_axis} by {color_col}")
            
            st.plotly_chart(fig)
        
        elif viz_type == "Scatter Plot":
            if len(numeric_cols) < 2:
                st.warning("Need at least 2 numeric columns for scatter plot")
                return
            
            x_axis = st.selectbox("Select X-axis column", numeric_cols)
            y_axis = st.selectbox("Select Y-axis column", numeric_cols)
            
            fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
            
            if len(numeric_cols) > 2:
                size_col = st.selectbox("Select size column (optional)", [None] + numeric_cols)
                color_col = st.selectbox("Select color column (optional)", [None] + categorical_cols)
                
                if size_col and color_col:
                    fig = px.scatter(df, x=x_axis, y=y_axis, size=size_col, color=color_col,
                                    title=f"{y_axis} vs {x_axis} (size: {size_col}, color: {color_col})")
                elif size_col:
                    fig = px.scatter(df, x=x_axis, y=y_axis, size=size_col,
                                    title=f"{y_axis} vs {x_axis} (size: {size_col})")
                elif color_col:
                    fig = px.scatter(df, x=x_axis, y=y_axis, color=color_col,
                                    title=f"{y_axis} vs {x_axis} (color: {color_col})")
            
            st.plotly_chart(fig)
        
        elif viz_type == "Histogram":
            if not numeric_cols:
                st.warning("No numeric columns found for histogram")
                return
            
            col = st.selectbox("Select column", numeric_cols)
            bins = st.slider("Number of bins", 5, 100, 20)
            
            fig = px.histogram(df, x=col, nbins=bins, title=f"Distribution of {col}")
            st.plotly_chart(fig)
        
        elif viz_type == "Box Plot":
            if not numeric_cols:
                st.warning("No numeric columns found for box plot")
                return
            
            y_axis = st.selectbox("Select value column", numeric_cols)
            x_axis = st.selectbox("Select category column (optional)", [None] + categorical_cols)
            
            if x_axis:
                fig = px.box(df, x=x_axis, y=y_axis, title=f"Distribution of {y_axis} by {x_axis}")
            else:
                fig = px.box(df, y=y_axis, title=f"Distribution of {y_axis}")
            
            st.plotly_chart(fig)
        
        elif viz_type == "Pie Chart":
            if not categorical_cols:
                st.warning("No categorical columns found for pie chart")
                return
            
            cat_col = st.selectbox("Select category column", categorical_cols)
            
            if numeric_cols:
                value_col = st.selectbox("Select value column", [None] + numeric_cols)
                if value_col:
                    fig = px.pie(df, names=cat_col, values=value_col, title=f"Distribution of {value_col} by {cat_col}")
                else:
                    fig = px.pie(df, names=cat_col, title=f"Distribution of {cat_col}")
            else:
                fig = px.pie(df, names=cat_col, title=f"Distribution of {cat_col}")
            
            st.plotly_chart(fig)
        
        elif viz_type == "Heatmap":
            if len(numeric_cols) < 2:
                st.warning("Need at least 2 numeric columns for heatmap")
                return
            
            cols = st.multiselect("Select numeric columns", numeric_cols, default=numeric_cols[:5])
            
            if len(cols) < 2:
                st.warning("Please select at least 2 columns")
                return
            
            corr = df[cols].corr()
            fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
            st.plotly_chart(fig)
        
        elif viz_type == "Violin Plot":
            if not numeric_cols:
                st.warning("No numeric columns found for violin plot")
                return
            
            y_axis = st.selectbox("Select value column", numeric_cols)
            x_axis = st.selectbox("Select category column (optional)", [None] + categorical_cols)
            
            if x_axis:
                fig = px.violin(df, x=x_axis, y=y_axis, box=True, title=f"Distribution of {y_axis} by {x_axis}")
            else:
                fig = px.violin(df, y=y_axis, box=True, title=f"Distribution of {y_axis}")
            
            st.plotly_chart(fig)
        
        elif viz_type == "Pair Plot":
            if len(numeric_cols) < 2:
                st.warning("Need at least 2 numeric columns for pair plot")
                return
            
            cols = st.multiselect("Select numeric columns", numeric_cols, default=numeric_cols[:5])
            
            if len(cols) < 2:
                st.warning("Please select at least 2 columns")
                return
            
            color_col = None
            if categorical_cols:
                color_col = st.selectbox("Select color column (optional)", [None] + categorical_cols)
            
            fig = px.scatter_matrix(df, dimensions=cols, color=color_col, title="Pair Plot")
            st.plotly_chart(fig)
    
    except Exception as e:
        st.error(f"Error creating visualization: {e}")

def main():
    st.sidebar.title("User Authentication")

    if 'email' not in st.session_state:
        st.session_state.email = ""
    
    if not st.session_state.email:
        action = st.sidebar.radio("Action", ["Login", "Register"], key="auth_action")
        email = st.sidebar.text_input("Email", "", key="auth_email")
        password = st.sidebar.text_input("Password", "", type="password", key="auth_password")

        if action == "Login":
            if st.sidebar.button("Login", key="login_button"):
                try:
                    user = auth.sign_in_with_email_and_password(email, password)
                    st.sidebar.success("Logged in successfully!")
                    st.session_state.email = email
                    st.session_state.chat_history = []
                except Exception as e:
                    st.sidebar.error(f"Login failed: {e}")
        elif action == "Register":
            if st.sidebar.button("Register", key="register_button"):
                try:
                    auth.create_user_with_email_and_password(email, password)
                    st.sidebar.success("User registered successfully!")
                except Exception as e:
                    st.sidebar.error(f"Registration failed: {e}")
    else:
        st.sidebar.write(f"Logged in as {st.session_state.email}")
        if st.sidebar.button("Logout", key="logout_button"):
            st.session_state.email = ""
            st.session_state.chat_history = []

        st.sidebar.title("Database Configuration")

        db_type = st.sidebar.selectbox("Select Database Type", ["MySQL", "PostgreSQL"], key="db_type")

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
            host = st.sidebar.text_input("Host", default_mysql["host"], key="mysql_host")
            port = st.sidebar.text_input("Port", default_mysql["port"], key="mysql_port")
            user = st.sidebar.text_input("Username", default_mysql["user"], key="mysql_user")
            password = st.sidebar.text_input("Database Password", default_mysql["password"], type="password", key="mysql_password")
            database = st.sidebar.text_input("Database", default_mysql["database"], key="mysql_database")
        elif db_type == "PostgreSQL":
            host = st.sidebar.text_input("Host", default_postgres["host"], key="pg_host")
            port = st.sidebar.text_input("Port", default_postgres["port"], key="pg_port")
            user = st.sidebar.text_input("Username", default_postgres["user"], key="pg_user")
            password = st.sidebar.text_input("Database Password", default_postgres["password"], type="password", key="pg_password")
            database = st.sidebar.text_input("Database", default_postgres["database"], key="pg_database")

        db_uri = f"{db_type.lower()}+mysqlconnector://{user}:{password}@{host}:{port}/{database}" if db_type == "MySQL" else f"{db_type.lower()}+psycopg2://{user}:{password}@{host}:{port}/{database}"

        if st.sidebar.button("Connect to Database", key="connect_db_button"):
            try:
                st.session_state.db, st.session_state.db_uri = init_database(db_type, user, password, host, port, database)
                st.success(f"Connected to {db_type} database successfully!")
            except Exception as e:
                st.error(f"Error connecting to database: {e}")

        st.sidebar.title("Create Database")
        with st.sidebar.expander("Create Database"):
            database_name = st.text_input("New Database Name", "", key="new_db_name")
            if st.button("Create Database", key="create_db_button"):
                if database_name:
                    create_database(db_type, user, password, host, port, database_name)
                else:
                    st.warning("Please enter a database name.")

    st.sidebar.title("Load Database")
    with st.sidebar.expander("Load Database"):
        uploaded_file = st.file_uploader("Choose a .sql file", type=["sql"], key="sql_uploader")
        if st.button("Load Database", key="load_db_button"):
            if uploaded_file is not None:
                if 'db' in st.session_state and st.session_state.db:
                    # Save uploaded file
                    temp_file = "temp_upload.sql"
                    with open(temp_file, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                    try:
                        # Connect to database
                        if db_type == "MySQL":
                            import mysql.connector
                            connection = mysql.connector.connect(
                                user=user, 
                                password=password, 
                                host=host, 
                                port=port, 
                                database=database,
                                autocommit=False
                            )
                        elif db_type == "PostgreSQL":
                            connection = psycopg2.connect(
                                user=user, 
                                password=password, 
                                host=host, 
                                port=port, 
                                database=database
                            )
                            connection.autocommit = False
                        else:
                            st.error("Unsupported database type")
                            return
                    
                    # Execute SQL file
                        success = execute_sql_file(connection, temp_file)
                    
                        if success:
                         # Clear any cached schema information
                            if 'db' in st.session_state:
                                del st.session_state.db
                            if 'db_uri' in st.session_state:
                                del st.session_state.db_uri
                        
                        # Reinitialize database connection
                            st.session_state.db, st.session_state.db_uri = init_database(
                                db_type, user, password, host, port, database
                            )
                        
                        # Clear any cached table info
                            if 'table_info' in st.session_state:
                                del st.session_state.table_info
                        
                            st.success("Database successfully updated! All functions will now use the new data.")
                        else:
                            st.warning("Database loaded with some errors. Some functions may not work correctly.")
                    
                    except Exception as e:
                        st.error(f"Error loading database: {str(e)}")
                    finally:
                        connection.close()
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                else:
                    st.error("Database not connected.")
            else:
                st.warning("Please upload a .sql file.")

    st.sidebar.title("Database Snapshots")
    snapshot_action = st.sidebar.selectbox("Action", ["Create Snapshot", "Restore Snapshot"], key="snapshot_action")

    if snapshot_action == "Create Snapshot":
        snapshot_name = st.sidebar.text_input("Snapshot Name", "snapshot.sql", key="snapshot_name")
        if st.sidebar.button("Create Snapshot", key="create_snapshot_button"):
            if 'db' in st.session_state and st.session_state.db:
                create_snapshot(
                    db_type=db_type,
                    user=user,
                    password=password,
                    host=host,
                    port=port,
                    database_name=database,
                    snapshot_path=snapshot_name
                )
            else:
                st.sidebar.error("Database not connected.")

    elif snapshot_action == "Restore Snapshot":
        snapshots = list_snapshots()
        if snapshots:
            snapshot_to_restore = st.sidebar.selectbox("Select Snapshot", snapshots, key="snapshot_select")
            if st.sidebar.button("Restore Snapshot", key="restore_snapshot_button"):
                if 'db' in st.session_state and st.session_state.db:
                    restore_snapshot(
                        db_uri=st.session_state.db_uri,
                        snapshot_name=snapshot_to_restore,
                        user=user,
                        password=password,
                        host=host,
                        port=port
                    )
                else:
                    st.sidebar.error("Database not connected.")
        else:
            st.sidebar.write("No snapshots available.")
            
    st.title("AI-Powered Data Query Interface")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Query Interface", "ER Diagram", "Logs", "Data Visualization", "Chat History"])

    with tab1:
        enable_tts = st.checkbox("Enable voice output", value=False, key="tts_checkbox")

        query = st.text_input("Enter your query:", key="query_input")
        submit_button = st.button("Submit", key="submit_button")

        if submit_button and query:
            if 'db' in st.session_state and st.session_state.db:
                response = get_response(query, st.session_state.db, [], db_type)
                st.write("Response:", response)
                
                # Save to chat history
                if st.session_state.email:
                    save_chat_history(st.session_state.email, query, response)
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    st.session_state.chat_history.append({"prompt": query, "response": response})
                
                if enable_tts:
                    speak_text(response)
            else:
                st.error("Database not connected.")

        if st.button("Speak", key="speak_button"):
            if 'db' in st.session_state and st.session_state.db:
                query = recognize_speech()
                st.text_input("Voice Query:", value=query, key="voice_query")
                if query:
                    response = get_response(query, st.session_state.db, [], db_type)
                    st.write("Response:", response)
                    
                    # Save to chat history
                    if st.session_state.email:
                        save_chat_history(st.session_state.email, query, response)
                        if 'chat_history' not in st.session_state:
                            st.session_state.chat_history = []
                        st.session_state.chat_history.append({"prompt": query, "response": response})
                    
                    if enable_tts:
                        speak_text(response)
            else:
                st.error("Database not connected.")

    with tab2:
        if st.button("Generate ER Diagram", key="er_diagram_button"):
            if 'db' in st.session_state and st.session_state.db:
                output_file = generate_er_diagram(st.session_state.db_uri)
                if output_file:
                    st.image(output_file)
            else:
                st.error("Database not connected.")

    with tab3:
        if st.session_state.email:
            export_chat_history(st.session_state.email)
        else:
            st.warning("Please log in to access logs.")

    with tab4:
        st.header("Interactive Data Visualization")
        
        if 'db' in st.session_state and st.session_state.db:
            # Get table information
            table_info = get_table_info(st.session_state.db_uri)
            
            if table_info:
                selected_table = st.selectbox("Select Table", list(table_info.keys()), key="table_select")
                
                if selected_table:
                    # Get data from the selected table
                    df = get_table_data(st.session_state.db_uri, selected_table)
                    
                    if df is not None and not df.empty:
                        # Show basic table info
                        st.write(f"### Table: {selected_table}")
                        st.write(f"Columns: {', '.join(table_info[selected_table]['columns'])}")
                        
                        # Show data preview
                        with st.expander("View Data Preview"):
                            st.dataframe(df.head())
                        
                        # Interactive visualization
                        visualize_data_interactive(df)
                    else:
                        st.warning("No data available for the selected table")
            else:
                st.warning("No tables found in the database")
        else:
            st.warning("Please connect to a database first")

    with tab5:
        st.header("Chat History")
        
        if st.session_state.email:
            # Load chat history for the current user
            chat_history_df = load_chat_history(st.session_state.email)
            
            if not chat_history_df.empty:
                st.write("### Your Conversation History")
                
                # Display chat history in a more conversational format
                for _, row in chat_history_df.iterrows():
                    with st.chat_message("user"):
                        st.write(f"**You ({row['DateTime']}):** {row['Prompt']}")
                    
                    with st.chat_message("assistant"):
                        st.write(f"**AI Assistant:** {row['Response']}")
                
                # Add buttons for managing chat history
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Export Chat History", key="export_chat_button"):
                        export_chat_history(st.session_state.email)
                with col2:
                    if st.button("Clear Chat History", key="clear_chat_button"):
                        clear_chat_history(st.session_state.email)
                        st.session_state.chat_history = []
                        st.experimental_rerun()
            else:
                st.info("No chat history found. Start a conversation in the Query Interface tab.")
        else:
            st.warning("Please log in to view your chat history.")

if __name__ == "__main__":
    main()

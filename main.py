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
from sqlalchemy import create_engine, inspect
from graphviz import Digraph
import pyrebase

# Load environment variables from .env file
load_dotenv()

# Ensure the API key is loaded
groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    st.error("GROQ_API_KEY is not set in the environment variables. Please check your .env file.")
    st.stop()

# Firebase configuration
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

# Initialize Pyrebase
firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

# Initialize TTS engine
tts_engine = pyttsx3.init()

def speak_text(text):
    def run_tts():
        try:
            tts_engine.setProperty('rate', 150)  # Set speech rate
            tts_engine.setProperty('volume', 1)  # Set volume
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            st.error(f"Error in TTS: {e}")

    # Run TTS in a separate thread
    tts_thread = threading.Thread(target=run_tts)
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
    
    # Pass the API key directly to ChatGroq
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

def generate_er_diagram(db_uri: str, output_file: str = "er_diagram"):
    try:
        engine = create_engine(db_uri)
        inspector = inspect(engine)
        
        dot = Digraph(format='png')

        # Add tables as nodes
        for table_name in inspector.get_table_names():
            dot.node(table_name, table_name)
            
            # Add columns as labels
            for column in inspector.get_columns(table_name):
                dot.node(f"{table_name}_{column['name']}", column['name'], shape='ellipse')
                dot.edge(table_name, f"{table_name}_{column['name']}")
                
        # Add relationships
        for table_name in inspector.get_table_names():
            foreign_keys = inspector.get_foreign_keys(table_name)
            for fk in foreign_keys:
                for column_name in fk['constrained_columns']:
                    # Add an edge from the current table to the referenced table
                    dot.edge(table_name, fk['referred_table'], label=column_name)
        
        # Render and save the diagram
        file_path = dot.render(filename=output_file, format='png', cleanup=False)
        if os.path.exists(file_path):
            return file_path
        else:
            st.error(f"File '{file_path}' not found.")
            return None
    except Exception as e:
        st.error(f"Error generating ER diagram: {e}")
        return None

def save_log(username, prompt, response):
    log_file_path = "logs/query_logs.csv"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = pd.DataFrame([{
        "Username": username,
        "DateTime": timestamp,
        "Prompt": prompt,
        "Response": response
    }])
    
    # Check if the log file already exists
    if os.path.exists(log_file_path):
        # Append to the existing file
        log_entry.to_csv(log_file_path, mode='a', header=False, index=False)
    else:
        # Create a new file with headers
        log_entry.to_csv(log_file_path, mode='w', header=True, index=False)

# Export logs function
def export_logs():
    log_file_path = "logs/query_logs.csv"
    if os.path.exists(log_file_path):
        with open(log_file_path, "rb") as file:
            btn = st.download_button(
                label="Download Logs",
                data=file,
                file_name="query_logs.csv",
                mime="text/csv",
            )
    else:
        st.warning("No logs available to download.")

# Set up Streamlit page
st.set_page_config(page_title="AI Powered Query Interface", page_icon=":robot:")

# Adding a logo and navigation bar
st.markdown(
    """
    <div>
        <h2>AI Powered Query Interface</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize session state variables
if "db" not in st.session_state:
    st.session_state["db"] = None

if "db_uri" not in st.session_state:
    st.session_state["db_uri"] = None

if "db_type" not in st.session_state:
    st.session_state["db_type"] = None

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "voice_output" not in st.session_state:
    st.session_state["voice_output"] = False

if "email" not in st.session_state:
    st.session_state["email"] = ""

# Sidebar content
with st.sidebar:
    st.title("Database Connection")
    
    db_type = st.selectbox("Select Database Type", ["MySQL", "PostgreSQL"], key="db_type")

    if db_type == "MySQL":
        st.text_input("MySQL Host", value="127.0.0.1", key="mysql_host")
        st.text_input("MySQL Port", value="3306", key="mysql_port")
        st.text_input("MySQL Username", value="root", key="mysql_user")
        st.text_input("MySQL Password", type="password", key="mysql_password")
        st.text_input("MySQL Database", key="mysql_database")
    elif db_type == "PostgreSQL":
        st.text_input("PostgreSQL Host", value="127.0.0.1", key="pgsql_host")
        st.text_input("PostgreSQL Port", value="5432", key="pgsql_port")
        st.text_input("PostgreSQL Username", key="pgsql_user")
        st.text_input("PostgreSQL Password", type="password", key="pgsql_password")
        st.text_input("PostgreSQL Database", key="pgsql_database")

    if st.button("Connect to Database"):
        try:
            if db_type == "MySQL":
                db, db_uri = init_database(
                    db_type="MySQL",
                    user=st.session_state.mysql_user,
                    password=st.session_state.mysql_password,
                    host=st.session_state.mysql_host,
                    port=st.session_state.mysql_port,
                    database=st.session_state.mysql_database
                )
            elif db_type == "PostgreSQL":
                db, db_uri = init_database(
                    db_type="PostgreSQL",
                    user=st.session_state.pgsql_user,
                    password=st.session_state.pgsql_password,
                    host=st.session_state.pgsql_host,
                    port=st.session_state.pgsql_port,
                    database=st.session_state.pgsql_database
                )
            
            st.session_state["db"] = db
            st.session_state["db_uri"] = db_uri
            st.success(f"Connected to {db_type} database successfully!")
        except Exception as e:
            st.error(f"Failed to connect to the database: {e}")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["Home", "Query Interface", "ER Diagram", "Export Logs"])

with tab1:
    st.header("Welcome to the AI Powered Query Interface")
    st.write("Use this application to interact with your database using natural language queries.")
    st.markdown(
"""
<div>
    <img src='https://thumbs.dreamstime.com/b/illustration-cloud-server-connected-to-database-miniature-analyst-analysis-data-data-center-concept-based-isometric-design-131817092.jpg' alt='Database Connection' style='width: 500px;'>
</div>
""",
 unsafe_allow_html=True
    )

with tab2:
    st.header("Query Interface")
    
    if not st.session_state["authenticated"]:
        username = st.text_input("Email", key="email")
        password = st.text_input("Password", type="password", key="password")
        
        if st.button("Login"):
            try:
                user = auth.sign_in_with_email_and_password(username, password)
                st.session_state["authenticated"] = True
                st.session_state["email"] = username
                st.success("Authentication successful!")
            except Exception as e:
                st.error(f"Authentication failed: {e}")
        
        if st.button("Register"):
            try:
                user = auth.create_user_with_email_and_password(username, password)
                st.success("Registration successful! You can now login.")
            except Exception as e:
                st.error(f"Registration failed: {e}")
            
    else:
        user_query = st.text_input("Enter your query", key="user_query")
        st.checkbox("Enable Voice Output", key="voice_output")

        if st.button("Submit Query"):
            if user_query and st.session_state.db:
                try:
                    db = st.session_state.db
                    db_type = st.session_state.db_type
                    chat_history = st.session_state.chat_history
                    
                    response = get_response(user_query, db, chat_history, db_type)
                    
                    st.session_state.chat_history.append(HumanMessage(content=user_query))
                    st.session_state.chat_history.append(AIMessage(content=response))
                    
                    st.write(response)
                    
                    save_log(st.session_state.email, user_query, response)
                    
                    if st.session_state.voice_output:
                        speak_text(response)
                except Exception as e:
                    st.error(f"Failed to process the query: {e}")
            else:
                st.warning("Please enter a query and ensure the database is connected.")

        if st.button("🎙 Speak"):
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                st.info("Listening...")
                audio = recognizer.listen(source)
                try:
                    user_query = recognizer.recognize_google(audio)
                    st.session_state.user_query = user_query
                    st.success(f"You said: {user_query}")
                except sr.UnknownValueError:
                    st.error("Google Speech Recognition could not understand audio")
                except sr.RequestError as e:
                    st.error(f"Could not request results from Google Speech Recognition service; {e}")

        if st.button("Logout"):
            st.session_state["authenticated"] = False
            st.success("Logged out successfully!")

with tab3:
    st.header("ER Diagram")
    if st.button("Generate ER Diagram"):
        db_uri = st.session_state.get("db_uri")
        if db_uri:
            file_path = generate_er_diagram(db_uri)
            if file_path:
                st.image(file_path, caption="Entity-Relationship Diagram")
        else:
            st.warning("Please connect to the database first.")

with tab4:
    st.header("Export Logs")
    export_logs()

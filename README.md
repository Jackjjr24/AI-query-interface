# AI-Powered Database Query Interface

![Project Banner](/images/banner.png)

An intelligent interface that allows users to interact with databases using natural language queries, powered by AI. The system supports MySQL and PostgreSQL databases, provides visualization tools, and includes user authentication.

## Features

### 🗣️ Natural Language Querying
- Convert natural language questions into SQL queries
- Get human-readable responses from database results
- Voice input/output support (speech-to-text and text-to-speech)

### 🔐 User Authentication
- Secure login/registration using Firebase Authentication
- Personalized chat history for each user
- Session management

### 🗄️ Database Management
- Connect to MySQL or PostgreSQL databases
- Create new databases
- Load SQL files to initialize/update databases
- Create and restore database snapshots

### 📊 Data Visualization
- Interactive charts and graphs (bar, line, scatter, histogram, etc.)
- Correlation heatmaps
- Pair plots for multivariate analysis
- Automatic detection of column types for appropriate visualizations

### 📝 ER Diagram Generation
- Visualize database schema
- Show tables, columns, and relationships
- Export diagram as PNG

### 📚 Chat History
- Save all queries and responses
- Export chat history as CSV
- Clear conversation history

## Technologies Used

- **Backend**: Python, LangChain, SQLAlchemy
- **AI**: Groq API (Mistral model)
- **Database**: MySQL, PostgreSQL
- **Authentication**: Firebase
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Voice**: Pyttsx3, SpeechRecognition
- **Frontend**: Streamlit
- **Other**: psycopg2, mysql-connector, graphviz

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Jackjjr24/AI-query-interface/
   cd AI-query-interface
   pip install -r requirements.txt
   streamlit run main.py

# AI-Powered Database Query Interface

![Project Banner](https://github.com/Jackjjr24/AI-query-interface/blob/main/images/banner)

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

2. **Create and activate a virtual environment**:

   ```bash
   python -m venv venv
   source venv\Scripts\activate

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   Set up environment variables:

4. **Create a .env file in the root directory**:

Add your Groq API key:

GROQ_API_KEY=your_api_key_here

5. **Install additional dependencies**:

For speech recognition on Linux:

      ```bash
      sudo apt-get install portaudio19-dev python3-pyaudio


For graphviz (ER diagrams):

      ```bash
         sudo apt-get install graphviz  # Linux
         brew install graphviz         # Mac

6. **Usage**:
Run the application:

      ```bash
         streamlit run app.py
         Access the interface:

The application will open in your default browser at http://localhost:8501

**Getting Started**:

   -Register or login using the sidebar

   -Configure your database connection

   -Start asking questions in natural language

**Configuration**

 **Database Connection**
   -Configure your database connection in the sidebar:

   -Select database type (MySQL or PostgreSQL)

   -Enter host, port, username, password, and database name

   -Click "Connect to Database"

**Voice Settings**
   -Enable/disable text-to-speech in the Query Interface tab

   -Use the "Speak" button for voice input

**Examples**
   
   -"Show me the top 10 customers by sales"

   "What's the average salary by department?"

   -"List all products with low inventory"

   -"Show me a correlation between age and spending"

**Visualization Examples**
   -Bar charts comparing categories

   -Scatter plots with trend lines

   -Pie charts for distribution analysis

   -Heatmaps for correlation visualization

**Screenshots**

![project image1](https://github.com/Jackjjr24/AI-query-interface/blob/main/images/image1)
![project image2](https://github.com/Jackjjr24/AI-query-interface/blob/main/images/image2)
![project image3](https://github.com/Jackjjr24/AI-query-interface/blob/main/images/image3)





# F.R.I.D.A.Y - AI-Powered Analytics Assistant

AI-powered analytics assistant for Google BigQuery. Ask questions in natural language, get SQL queries, and visualize results.

## 🎯 Features

- **Natural Language to SQL**: Ask questions about your data in plain English
- **Multi-Table Context**: Work with multiple BigQuery tables simultaneously
- **AI-Powered Descriptions**: Automatically generate table and column descriptions
- **Interactive Planning**: Iteratively refine query plans with AI assistance
- **Smart Visualizations**: Automatic chart type suggestions based on query results
- **Context Persistence**: Your table contexts are saved between sessions

## 🏗️ Architecture

This application has been refactored into a modular structure:

```
GenSQL/
├── src/
│   ├── config/           # Configuration and settings
│   ├── database/         # BigQuery and ChromaDB clients
│   ├── ai/              # Vertex AI, embeddings, SQL generation
│   ├── context/         # Context management and persistence
│   ├── ui/              # UI components (in progress)
│   └── utils/           # Utility functions
├── tests/               # Unit and integration tests
├── data/                # Sample data files
├── app.py               # Main application entry point
├── requirements.txt     # Python dependencies
└── .env.example         # Environment variable template

```

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Google Cloud Platform account with:
  - BigQuery API enabled
  - Vertex AI API enabled
  - Application Default Credentials configured

### Installation

1. **Clone the repository**
   ```bash
   cd GenSQL
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   copy .env.example .env
   # Edit .env with your configuration
   ```

5. **Authenticate with Google Cloud**
   ```bash
   gcloud auth login
   gcloud auth application-default login
   gcloud config set project YOUR_PROJECT_ID
   ```

### Running the App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

## 📖 Usage

### 1. Set Context

Choose tables to analyze:
- **Upload Schema**: Upload a CSV file with table schema
- **Pick from BigQuery**: Browse your BigQuery project and select tables

### 2. Describe Your Tables

- Add table descriptions
- Auto-generate column descriptions using AI
- Edit descriptions to improve query quality

### 3. Ask Questions

Type your question in natural language:
- "What were the top 10 stations by trip count?"
- "Show me the average trip duration by month"
- "Which day of the week has the most trips?"

### 4. Refine the Plan

- Review the AI-generated query plan
- Suggest modifications through natural conversation
- Approve when satisfied

### 5. Execute & Visualize

- Run the generated SQL on BigQuery
- View results in a table
- Create charts when applicable

## 🔧 Configuration

Environment variables (`.env` file):

```bash
# GCP Configuration
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1
VERTEX_LOCATION=us-central1

# Model Configuration
GEMINI_MODEL=gemini-2.5-flash
EMBEDDING_MODEL_PRIMARY=text-embedding-005

# Application Settings
ENVIRONMENT=development
LOG_LEVEL=INFO
```

## 🧪 Development

### Code Structure

- **`src/config`**: Environment-based configuration
- **`src/database`**: Database clients with caching
- **`src/ai`**: LLM interactions (Gemini, embeddings)
- **`src/context`**: Context and state management
- **`src/ui`**: Streamlit UI components

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
flake8 src/ tests/

# Type checking
mypy src/
```

## 📝 Refactoring Status

### ✅ Completed
- Configuration module with environment variables
- BigQuery client with caching
- ChromaDB client for vector storage
- Vertex AI initialization and model management
- Embedding generation
- SQL and plan generation
- Column description enhancement
- Context management (save/load/search)
- State persistence
- Session state initialization

### 🚧 In Progress
- UI page extraction to `src/ui/pages/`
- Sidebar component extraction
- Helper function utilities

### 📋 Planned
- Unit test suite
- Integration tests
- Docker containerization
- CI/CD pipeline
- Production deployment guide

## 🔐 Security

- Never commit `.env` files or credentials
- Use Application Default Credentials (ADC) for GCP authentication
- Enable audit logging for production deployments
- Review the [productionization plan](implementation_plan.md) for security best practices

## 📄 License

This project is for internal use.

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Add tests
4. Submit a pull request

## 📞 Support

For questions or issues, contact the development team.

---

**Note:** This application is currently being refactored from a monolithic structure to a modular architecture. See [task.md](task.md) for refactoring progress.

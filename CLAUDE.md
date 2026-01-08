# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

F.R.I.D.A.Y is an AI-powered analytics assistant that converts natural language questions into BigQuery SQL queries. Users select BigQuery tables, describe them, ask questions in natural language, and get generated SQL with visualizations.

## Commands

```bash
# Run the application
streamlit run app_new.py

# Run tests
pytest tests/

# Code formatting and linting
black src/ tests/
flake8 src/ tests/
mypy src/
```

## Architecture

### Core Flow
1. **Context Setup**: User selects BigQuery tables and provides/generates column descriptions
2. **Question Processing**: Natural language question → Query plan → User approval → SQL generation
3. **Execution**: SQL runs on BigQuery, results displayed with auto-suggested charts

### Module Structure

**`src/config/settings.py`** - Singleton `Settings` class loading from environment variables. Import as `from src.config import settings`.

**`src/database/`**
- `bigquery_client.py` - Cached BQ client (`get_bq_client()`), schema fetching, query execution
- `chroma_client.py` - In-memory vector store using numpy cosine similarity for semantic search (replaced ChromaDB)

**`src/ai/`**
- `vertex_client.py` - Cached Vertex AI initialization, Gemini model (`get_model()`), embedding model
- `sql_generator.py` - Plan generation, SQL generation from approved plans, chart suggestions
- `embeddings.py` - Embedding generation wrapper
- `column_enhancer.py` - AI-generated column descriptions

**`src/context/`**
- `manager.py` - Table/column context CRUD with vector embeddings, multi-table support
- `persistence.py` - Session state persistence across app reloads

**`src/ui/`**
- `state.py` - Session state initialization (`init_session_state()`)
- `pages/` - Streamlit page renderers (context_page, view_context, analysis)
- `components/` - Reusable UI components (sidebar)

### Key Patterns

**Session State**: All state is managed via `st.session_state`. Key variables:
- `selected_tables` - List of table info dicts (project, dataset, table, description)
- `schemas` - Dict mapping table FQN to schema DataFrames
- `messages` - Chat history
- `ctx_set` - Boolean flag indicating if context is ready

**Caching**: Uses `@st.cache_resource` for clients/models, `@st.cache_data` for data fetching.

**Vector Search**: Embeddings stored in-memory with numpy-based cosine similarity. Format mimics ChromaDB for API compatibility.

## GCP Dependencies

Requires authenticated Google Cloud SDK:
```bash
gcloud auth login
gcloud auth application-default login
```

APIs needed: BigQuery, Vertex AI

## Environment Variables

Copy `.env.example` to `.env`. Key settings:
- `GCP_PROJECT_ID` - Optional, auto-detected from ADC if not set
- `GEMINI_MODEL` - Default: gemini-2.5-flash
- `VERTEX_LOCATION` - Default: us-central1

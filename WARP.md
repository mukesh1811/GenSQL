# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project overview

F.R.I.D.A.Y is a Streamlit-based, AI-powered analytics assistant for Google BigQuery. Users:
- Set context by uploading a table schema or selecting BigQuery tables
- Use a chat-style interface to iteratively design an analysis plan
- Generate and run BigQuery SQL
- Optionally visualize and summarize results

The repo has been refactored from a single monolithic `app.py` into a modular `src/` package plus a thin Streamlit entry point.

Key docs to be aware of:
- `README.md` – main setup, architecture, and development guide
- `MIGRATION.md` – how to move between the legacy monolith and the refactored modular app
- `SECRETS_SETUP.md` – recommended production secrets management on GCP
- `scope.md` – product/vision context

## Core workflows & commands

All commands assume the working directory is the repo root (`GenSQL/`). On Windows/PowerShell, keep using `python` as in these examples.

### Environment setup

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
# Install dependencies
pip install -r requirements.txt
```

The app reads configuration from `.env`; start from the template:

```bash
copy .env.example .env
# then edit .env with your project-specific values
```

For Google Cloud authentication (local development):

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### Running the app (refactored version)

The primary entry point for the refactored modular app is `app_new.py`:

```bash
streamlit run app_new.py
```

`MIGRATION.md` documents how to rename/swap `app_new.py` with a top-level `app.py` if you want the new version to be the default (and optionally keep the legacy `archive/app.py` around as a fallback). Do not assume `app.py` exists; check the current state before using `streamlit run app.py`.

### Tests

Tests use `pytest` (see `README.md`). To run the whole test suite:

```bash
pytest tests/
```

To run a single test file or test (standard `pytest` usage):

```bash
# single file
pytest tests/test_some_module.py

# single test by node id
pytest tests/test_some_module.py::test_case_name
```

### Code quality tools

The project expects standard Python tooling (you may need to install these as dev dependencies if they are not already present):

```bash
# Formatting
black src/ tests/

# Linting
flake8 src/ tests/

# Type checking
mypy src/
```

### BigQuery schema helper

To obtain a schema CSV for the “Upload Schema” flow, `src/utils/constants.py` exposes a canonical INFORMATION_SCHEMA query (also shown in the Context UI). In BigQuery, run a variant of:

```sql
SELECT
  table_catalog,
  table_schema,
  table_name,
  column_name,
  data_type,
  column_description
FROM `<your-project-id>.<your_dataset_id>.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = '<your_table_name>';
```

Save the result as CSV and upload it in the app’s "Upload Schema" tab.

## High-level architecture

### Entry point & UI routing

- **`app_new.py`** is the main Streamlit entry point for the refactored app.
  - Configures the page using `src.config.settings.settings` (title, icon, environment flags).
  - Calls `src.ui.state.init_session_state()` to ensure all `st.session_state` keys are initialized.
  - Calls `src.context.persistence.load_persisted_context_to_session()` to restore any prior context within the same session.
  - Renders the sidebar via `src.ui.components.sidebar.render_sidebar()`.
  - Routes based on `st.session_state.view` to one of:
    - `src.ui.pages.context_page.render_ctx_page()` – context setup
    - `src.ui.pages.view_context.render_view_context_page()` – context inspection
    - `src.ui.pages.analysis.render_analysis_page()` – main chat/analysis experience (default)

The legacy monolithic implementation lives under `archive/app.py` and is described in `MIGRATION.md`; new work should target the modular structure unless you are explicitly doing migration work.

### Configuration & settings layer

- **`src/config/settings.py`** exposes a singleton `settings` object that reads all configuration from environment variables (or sensible defaults):
  - GCP & Vertex AI: `GCP_PROJECT_ID`, `GCP_REGION`, `VERTEX_LOCATION`
  - Model IDs: `GEMINI_MODEL`, `GEMINI_FALLBACK_MODEL`, `EMBEDDING_MODEL_PRIMARY`, `EMBEDDING_MODEL_FALLBACK`
  - App meta: `APP_NAME`, `APP_ICON`, `ENVIRONMENT`, `LOG_LEVEL`
  - Feature flags & BigQuery cost estimation: `ENABLE_ANALYTICS`, `ENABLE_COST_TRACKING`, `CACHE_TTL`, `BQ_COST_PER_TB`
- This module is used across the codebase (e.g., in `src.ai.vertex_client`) to keep configuration centralized and environment-driven.

### Data access & in-memory context storage

- **`src/database/bigquery_client.py`**
  - Provides a cached `bigquery.Client` (`get_bq_client`) using Application Default Credentials.
  - Surface helpers for:
    - Project/dataset/table discovery: `list_projects`, `list_datasets`, `list_tables`
    - Schema retrieval: `get_table_schema(project, dataset, table)` returning a `pandas.DataFrame` with standard schema columns plus an editable `column_description` column. If descriptions have been stored in the vector store, they are merged back in via `src.context.manager.get_context`.
    - Query execution: `run_query(sql_query)` returning a `DataFrame` with error reporting through Streamlit.
    - Cost estimation: `estimate_query_cost(sql_query)` using a BigQuery dry run and `settings.BQ_COST_PER_TB`.

- **`src/database/chroma_client.py`**
  - Implements an in-memory, session-scoped “vector store” using `st.session_state.vector_store`.
  - Provides simple collections:
    - `schema_context` – embeddings plus metadata for table/column descriptions.
    - `app_state` – general app-level markers (e.g., the current context selection).
  - Includes `search_by_embedding` and `cosine_similarity` using NumPy; the public API is intentionally shaped like ChromaDB’s query response for backward compatibility.

This layer is responsible for both talking to external data sources (BigQuery) and maintaining a search-friendly representation of table/column context in memory.

### AI integration layer

- **`src/ai/vertex_client.py`**
  - Centralizes initialization of Vertex AI using the same project as the BigQuery client (`bigquery.Client().project`) or `settings.GCP_PROJECT_ID` when provided.
  - Exposes cached:
    - `get_model()` – generative Gemini model for chat/plan/SQL generation, with fallback to a secondary model ID.
    - `get_embedding_model()` – text embedding model with a primary/fallback sequence; does a small warmup call to fail fast.

- **`src/ai/embeddings.py`**
  - Thin wrapper `get_embedding(text)` that calls `get_embedding_model()` and returns the embedding vector for a single string.

- **`src/ai/sql_generator.py`**
  - Builds the **LLM context** by combining all selected tables (`st.session_state.selected_tables`) and their schemas (`st.session_state.schemas`), with fallback to the legacy single-table `schema_df`.
  - Core responsibilities:
    - `generate_plan(conversation_text)` – produces a natural-language analysis plan (goal + numbered steps) given conversation history and table context.
    - `generate_sql_from_plan(conversation_text, approved_plan)` – turns an approved plan into a single BigQuery SQL query string (or a sentinel string if unanswerable).
    - `suggest_chart(user_prompt, sql_query, result_df)` – chooses a chart type and axes based on user intent and result data, returning a small JSON-like dict describing the visualization.
    - `summarize_query_result(result_df, user_question)` – uses the LLM (with heuristics fallback) to produce a short, human-readable insight from the query results.

- **`src/ai/column_enhancer.py`**
  - Uses Vertex AI plus lightweight BigQuery statistics to:
    - Generate/augment table descriptions (`enhance_table_description`).
    - Generate per-column descriptions (`enhance_column_descriptions`) by combining type-aware summary queries and LLM output, with heuristic fallbacks if parsing fails.

Collectively, this layer owns all LLM interactions: plan refinement, SQL generation, schema/column description enhancement, chart suggestion, and query-result summarization.

### Context management & persistence

- **`src/context/manager.py`**
  - Sits between the UI and the in-memory vector store, managing the notion of “context” (selected tables + schemas + descriptions):
    - `save_context`, `persist_schema_to_chroma` – store column-level descriptions as embeddings in the `schema_context` collection.
    - `get_context(table_id=None, column_name=None)` – retrieve stored metadata/embeddings for semantic augmentation.
    - `search_similar_contexts(query, n_results)` – semantic search over stored context using embeddings.
    - `set_context(...)`, `add_table_to_context(...)`, `remove_table_from_context(...)`, `clear_context()` – manage `st.session_state.selected_tables`, `schemas`, `schema_df`, and keep the UI in sync.
    - `set_ctx_if_ready()` – computes the boolean `st.session_state.ctx_set` that gates most of the analysis flows.

- **`src/context/persistence.py`**
  - Implements **session-scoped persistence** of the current context:
    - `persist_current_context_marker()` – stores a high-level “current context” doc and metadata in the `app_state` collection (as JSON describing selected tables).
    - `load_persisted_context_to_session()` – on app startup (within the same Streamlit session), restores `selected_tables` and schemas either from BigQuery or, if necessary, from the in-memory vector store.

This layer provides a consistent, serializable representation of “what the user is analyzing” and keeps it resilient to partial reloads while explicitly not persisting across full app restarts.

### UI: pages & components

- **`src/ui/state.py`**
  - Central place for initializing all `st.session_state` keys used across the app (view routing, auth flags, selected tables, schemas, per-tab buffers, message history, flags like `current_plan_approved`, etc.).

- **`src/ui/components/sidebar.py`**
  - Renders the left sidebar used in `app_new.py`.
  - Shows high-level context status (empty vs. set), quick navigation buttons (Set/Change/View Context), and an inline “How it works” walkthrough of the end-to-end flow.

- **`src/ui/pages/context_page.py`**
  - Implements the **“Set Context”** experience via two tabs:
    - **Upload Schema**: lets users upload a schema CSV or load a sample schema, edit column descriptions in a data editor, optionally auto-enhance descriptions, and then set/append this table as context.
    - **Pick a BigQuery table**: guides users through GCP login (conceptual; actual auth is handled by ADC), project/dataset/table selection using helpers from `src.database.bigquery_client`, and editing/enhancing table/column descriptions before setting/appending context.
  - Also presents a “Current Context” section listing all tables in context with the ability to remove individual tables or clear all context, plus navigation back to the main analysis view.

- **`src/ui/pages/analysis.py`**
  - Hosts the **chat-style analysis workflow**:
    - Enforces that context is set (`ctx_set`) before allowing input.
    - Renders the full message history from `st.session_state.messages`, including:
      - User messages
      - Assistant “plan” messages (goal + steps)
      - Assistant “sql” messages (generated SQL, query results, optional summaries, and optional charts)
    - Provides controls to:
      - Approve the latest unapproved plan and trigger SQL generation.
      - Run generated SQL via `src.database.run_query` and display truncated/full results.
      - Summarize results using `summarize_query_result`.
      - Request and render charts suggested by `suggest_chart`.
      - Manually edit the generated SQL before executing.
    - On each new user prompt, builds an LLM-ready conversation transcript via `src.utils.chat.build_llm_conversation_text` and calls `generate_plan`.

- **`src/ui/pages/view_context.py`**
  - Read-only view of the current context: iterates through `selected_tables`, shows descriptions, and displays the associated schema DataFrame (with a legacy fallback to `schema_df`).
  - Provides navigation back to analysis or to context modification.

The UI layer is responsible for orchestrating the user journey from context setup → chat-driven planning → SQL generation/execution → summarization/visualization, relying heavily on the context, AI, and database layers described above.

### Utilities & supporting modules

- **`src/utils/chat.py`** – formats the chat history into a simple, role-prefixed transcript string for use in LLM prompts.
- **`src/utils/constants.py`** – holds the canonical BigQuery schema extraction query used in the UI.
- **`tests/`** – Python package placeholder for unit/integration tests (currently minimal; see `README.md` for testing plans).

## Migration notes for agents

- If you need to touch or compare the legacy implementation, use `archive/app.py` but treat it as read-only reference unless you are deliberately doing migration work.
- `MIGRATION.md` documents how the old monolithic responsibilities map into the new modules (`src/config/settings.py`, `src/database/*`, `src/ai/*`, `src/context/*`, `src/ui/*`). When adding new features, prefer extending the modular structure rather than re-introducing logic into the entry point.
- When adjusting anything around authentication, secrets, or deployment, consult `SECRETS_SETUP.md` and the environment variables listed in `README.md`/`.env.example` instead of hardcoding values.

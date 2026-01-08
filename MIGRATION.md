# Refactoring Migration Guide

This guide helps you transition from the monolithic `app.py` to the refactored modular structure.

## Quick Start

### Option 1: Keep Both Versions (Recommended for Testing)

1. **Rename the original app**
   ```bash
   mv app.py app_legacy.py
   ```

2. **Use the new modular version**
   ```bash
   mv app_new.py app.py
   ```

3. **Test the new version**
   ```bash
   streamlit run app.py
   ```

4. **If issues occur, revert to legacy**
   ```bash
   streamlit run app_legacy.py
   ```

### Option 2: Direct Migration

If you're confident in the refactored code:

1. **Backup the original**
   ```bash
   cp app.py app_legacy.py
   ```

2. **Replace with new version**
   ```bash
   cp app_new.py app.py
   ```

## What's Changed

### Before (Monolithic)
```
app.py (1672 lines)
├── Configuration (hardcoded)
├── Database clients
├── AI/LLM logic
├── Context management
├── UI components
└── Helper functions
```

### After (Modular)
```
src/
├── config/settings.py (Configuration)
├── database/
│   ├── bigquery_client.py
│   └── chroma_client.py
├── ai/
│   ├── vertex_client.py
│   ├── embeddings.py
│   ├── sql_generator.py
│   └── column_enhancer.py
├── context/
│   ├── manager.py
│   └── persistence.py
└── ui/
    └── state.py

app.py (minimal entry point)
```

## Import Changes

### Old Way
```python
# Everything in one file
def get_bq_client():
    return bigquery.Client()

client = get_bq_client()
```

### New Way
```python
# Import from modules
from src.database import get_bq_client

client = get_bq_client()
```

## Configuration Migration

### Old: Hardcoded
```python
VERTEX_LOCATION = "us-central1"
GEMINI_MODEL = "gemini-2.5-flash"
```

### New: Environment Variables
```python
# .env file
VERTEX_LOCATION=us-central1
GEMINI_MODEL=gemini-2.5-flash

# In code
from src.config import settings
location = settings.VERTEX_LOCATION
```

## Testing the Migration

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment**
   ```bash
   cp .env.example .env
   # Edit .env with your values
   ```

3. **Test database connections**
   ```python
   from src.database import get_bq_client, get_chroma_client
   
   bq = get_bq_client()
   chroma = get_chroma_client()
   ```

4. **Test AI modules**
   ```python
   from src.ai import get_model, get_embedding
   
   model = get_model()
   embedding = get_embedding("test")
   ```

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'src'`

**Solution:** Make sure you're running from the GenSQL directory:
```bash
cd m:\GenSQL
streamlit run app.py
```

### ChromaDB Path Issues

**Problem:** ChromaDB can't find the database

**Solution:** Check `.env` file:
```bash
CHROMA_PERSIST_DIR=./chroma_db
```

### Vertex AI Authentication

**Problem:** `Could not automatically determine credentials`

**Solution:**
```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

## Rollback Procedure

If you need to revert to the original:

```bash
# Stop the app (Ctrl+C)
streamlit run app_legacy.py
```

## Next Steps

Once comfortable with the refactored version:

1. ✅ **Extract remaining UI pages** to `src/ui/pages/`
2. ✅ **Add unit tests** in `tests/`
3. ✅ **Set up CI/CD** with GitHub Actions
4. ✅ **Deploy to Cloud Run** (see implementation_plan.md)

## Getting Help

- Check `README.md` for general documentation
- Review `implementation_plan.md` for productionization details
- Examine `task.md` for refactoring progress

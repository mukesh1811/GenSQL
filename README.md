# F.R.I.D.A.Y Production Stack

This repository hosts a production-ready implementation of the FRIDAY analytics assistant.

## Structure

```
apps/
├── backend   # FastAPI service exposing analytics APIs
└── frontend  # React/Vite dashboard for analysts
```

## Getting started

### Backend

```bash
cd apps/backend
pip install -e .
uvicorn app.main:app --reload
```

The service expects Google Cloud Application Default Credentials with access to BigQuery and Vertex AI.

### Frontend

```bash
cd apps/frontend
npm install
npm run dev
```

The Vite dev server proxies API requests to `http://localhost:8000` by default.

## Key capabilities

- Persist BigQuery table context and column metadata via ChromaDB
- Generate Vertex AI powered analysis plans and SQL
- Execute queries in BigQuery and visualize results
- Summarize result sets and propose charts automatically


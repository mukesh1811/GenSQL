# Logic for Secrets Management

This guide explains how to secure your sensitive configuration (API keys, credentials) using **Google Secret Manager**, which is the recommended approach for production on Google Cloud.

## 1. Prerequisites

Ensure you have the Google Cloud CLI installed and authorized:

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

## 2. Enable Secret Manager API

Enable the service in your project:

```bash
gcloud services enable secretmanager.googleapis.com
```

## 3. Create Secrets

Create secrets for your sensitive variables. For example, if you have an API key or database password:

```bash
# Create the secret
gcloud secrets create GEMINI_API_KEY --replication-policy="automatic"

# Add a secret version (the actual value)
echo -n "your-super-secret-key-value" | gcloud secrets versions add GEMINI_API_KEY --data-file=-
```

Repeat this for other sensitive variables like `DB_PASSWORD`, `SERVICE_ACCOUNT_KEY`, etc.

## 4. Access Secrets in Cloud Run

When deploying to Cloud Run, you don't need to change your Python code. You map the secret to an environment variable.

**Deploy command example:**

```bash
gcloud run deploy friday-app \
  --image gcr.io/YOUR_PROJECT_ID/friday-app \
  --set-secrets="GEMINI_API_KEY=GEMINI_API_KEY:latest" \
  --region us-central1
```

Inside your app (`src/config/settings.py`):
```python
import os
# This continues to work because Cloud Run injects the secret as an env var!
api_key = os.getenv("GEMINI_API_KEY") 
```

## 5. Local Development (Hybrid)

For testing locally, use a `.env` file (ensure strict `.gitignore`).

1. **Create .env:**
   ```bash
   cp .env.example .env
   ```
2. **Edit .env** and add your actual keys.
3. **Run App:** `streamlit run app.py`

Your app is configured to use `deployment` environment variables if present, which makes switching between local `.env` and Cloud Run Secrets seamless.

# Week 6: Agent Deployment (Final Week)
[back](README.md)
## Student Reading Material

### Introduction

Welcome to Week 6, the final week of the Agentic AI program. This week focuses on taking your AI agent from development to production. You'll learn how to deploy agents so they're accessible via browser, manage secrets securely, and follow best practices for production systems.

**Goal:** Deploy a working AI agent accessible via browser.

### What is Agent Deployment?

Deployment is the process of making your AI agent available for real users. This involves:

- **Containerization**: Packaging your agent with all dependencies
- **Hosting**: Running your agent on cloud infrastructure
- **API Exposure**: Making your agent accessible via HTTP endpoints
- **Security**: Managing API keys and secrets safely
- **Monitoring**: Tracking performance and errors
- **Version Control**: Managing code changes and releases

### Learning Outcomes

By the end of this week, you will be able to:
- Containerize AI agents using Docker
- Deploy agents using Docker Compose
- Deploy agents to Google Vertex AI
- Manage API keys and secrets securely
- Use Git for version control
- Implement deployment best practices
- Create browser-accessible agent interfaces
- Monitor and debug production agents

---

## Deployment Options

### Option 1: Docker Compose (Local/VPS)

Docker Compose allows you to define and run multi-container applications. Ideal for local development and VPS deployment.

**Advantages:**
- Full control over infrastructure
- Cost-effective for small to medium scale
- Easy local development
- Portable across environments

**Use Cases:**
- Development and testing
- Small production deployments
- Self-hosted solutions
- VPS deployments (DigitalOcean, Linode, AWS EC2)

### Option 2: Google Vertex AI

Vertex AI is Google Cloud's managed ML platform with built-in support for deploying AI models and agents.

**Advantages:**
- Fully managed infrastructure
- Auto-scaling
- Built-in monitoring
- Integration with Google Cloud services
- Enterprise-grade security

**Use Cases:**
- Production deployments
- High-traffic applications
- Enterprise solutions
- Teams needing managed infrastructure

---

## Docker Deployment

### Understanding Docker

Docker packages your application and all dependencies into a container that runs consistently anywhere.

**Key Concepts:**

**1. Dockerfile**: Instructions to build your container

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "app.py"]
```

**2. Docker Image**: Built package containing your application

```bash
docker build -t my-agent:latest .
```

**3. Docker Container**: Running instance of an image

```bash
docker run -p 8000:8000 my-agent:latest
```

### Building an Agent API

Create a web API for your agent using FastAPI:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="AI Agent API")

class Query(BaseModel):
    question: str
    session_id: str = None

class Response(BaseModel):
    answer: str
    sources: list = []
    session_id: str

# Initialize your agent
from agent import MultiToolAgent
agent = MultiToolAgent()

@app.get("/")
def root():
    return {"message": "AI Agent API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/query", response_model=Response)
def query_agent(query: Query):
    try:
        result = agent.run(query.question)
        return Response(
            answer=result["answer"],
            sources=result.get("sources", []),
            session_id=query.session_id or "default"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tools")
def list_tools():
    return {"tools": agent.list_tools()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Complete Dockerfile

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for models and data
RUN mkdir -p /app/models /app/data

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose Setup

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  agent:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ENVIRONMENT=production
    volumes:
      - ./models:/app/models
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - agent
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
```

### Nginx Configuration

Create `nginx.conf` for reverse proxy:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream agent {
        server agent:8000;
    }

    server {
        listen 80;
        server_name your-domain.com;

        location / {
            proxy_pass http://agent;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /health {
            proxy_pass http://agent/health;
        }
    }
}
```

### Deployment Commands

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f agent

# Stop services
docker-compose down

# Rebuild after code changes
docker-compose up -d --build

# Scale agent service
docker-compose up -d --scale agent=3
```

---

## Google Vertex AI Deployment

### Setting Up Vertex AI

**Step 1: Install Google Cloud SDK**

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize
gcloud init

# Set project
gcloud config set project YOUR_PROJECT_ID
```

**Step 2: Enable Required APIs**

```bash
gcloud services enable \
    aiplatform.googleapis.com \
    cloudbuild.googleapis.com \
    run.googleapis.com
```

**Step 3: Authenticate**

```bash
gcloud auth login
gcloud auth application-default login
```

### Containerize for Vertex AI

Create `Dockerfile` optimized for Vertex AI:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Port for Cloud Run
ENV PORT=8080
EXPOSE 8080

# Run with gunicorn for production
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
```

### Deploy to Cloud Run (via Vertex AI)

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/agent

# Deploy to Cloud Run
gcloud run deploy agent \
    --image gcr.io/YOUR_PROJECT_ID/agent \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --set-env-vars OPENAI_API_KEY=your-key \
    --memory 2Gi \
    --cpu 2 \
    --max-instances 10
```

### Vertex AI Agent Builder

Use Vertex AI's Agent Builder for managed deployment:

```python
from google.cloud import aiplatform

# Initialize Vertex AI
aiplatform.init(project="YOUR_PROJECT_ID", location="us-central1")

# Deploy model endpoint
endpoint = aiplatform.Endpoint.create(display_name="agent-endpoint")

# Deploy your agent
model = aiplatform.Model.upload(
    display_name="my-agent",
    artifact_uri="gs://your-bucket/agent",
    serving_container_image_uri="gcr.io/YOUR_PROJECT_ID/agent"
)

model.deploy(
    endpoint=endpoint,
    machine_type="n1-standard-4",
    min_replica_count=1,
    max_replica_count=5
)
```

### Vertex AI Configuration File

Create `vertex_config.yaml`:

```yaml
apiVersion: v1
kind: Agent
metadata:
  name: my-ai-agent
spec:
  runtime: python3.11
  resources:
    memory: 2Gi
    cpu: 2
  scaling:
    minReplicas: 1
    maxReplicas: 10
  environment:
    - name: OPENAI_API_KEY
      valueFrom:
        secretKeyRef:
          name: openai-key
          key: api-key
  healthCheck:
    path: /health
    initialDelaySeconds: 30
    periodSeconds: 10
```

---

## Managing API Keys & Secrets

### Never Commit Secrets to Git

**Bad Practice:**
```python
# DON'T DO THIS
OPENAI_API_KEY = "sk-abc123..."
```

**Good Practice:**
```python
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

### Environment Variables

Create `.env` file (add to `.gitignore`):

```bash
# .env
OPENAI_API_KEY=sk-abc123...
DATABASE_URL=postgresql://user:pass@localhost/db
REDIS_URL=redis://localhost:6379
SECRET_KEY=your-secret-key
```

Load in Python:

```python
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
```

### Docker Secrets

Use environment variables in Docker Compose:

```yaml
services:
  agent:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    env_file:
      - .env
```

Run with:
```bash
docker-compose --env-file .env up
```

### Google Cloud Secret Manager

Store secrets securely in Google Cloud:

```bash
# Create secret
echo -n "sk-abc123..." | gcloud secrets create openai-key --data-file=-

# Grant access
gcloud secrets add-iam-policy-binding openai-key \
    --member="serviceAccount:YOUR_SERVICE_ACCOUNT" \
    --role="roles/secretmanager.secretAccessor"
```

Access in Python:

```python
from google.cloud import secretmanager

def get_secret(secret_id, project_id):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

OPENAI_API_KEY = get_secret("openai-key", "YOUR_PROJECT_ID")
```

### AWS Secrets Manager

For AWS deployments:

```python
import boto3
import json

def get_secret(secret_name, region_name="us-east-1"):
    client = boto3.client('secretsmanager', region_name=region_name)
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

secrets = get_secret("agent-secrets")
OPENAI_API_KEY = secrets["OPENAI_API_KEY"]
```

---

## Version Control with Git

### Git Basics

**Initialize Repository:**

```bash
git init
git add .
git commit -m "Initial commit"
```

**Essential Commands:**

```bash
# Check status
git status

# Add files
git add filename.py
git add .

# Commit changes
git commit -m "Add new feature"

# View history
git log

# Create branch
git checkout -b feature-branch

# Merge branch
git checkout main
git merge feature-branch

# Push to remote
git push origin main
```

### .gitignore File

Create `.gitignore` to exclude sensitive files:

```
# Environment variables
.env
.env.local
.env.production

# API keys
**/secrets/
*.key
*.pem

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/

# Models and data
*.pkl
*.h5
*.pt
models/
data/

# Logs
*.log
logs/

# IDE
.vscode/
.idea/
*.swp

# Docker
docker-compose.override.yml

# OS
.DS_Store
Thumbs.db
```

### Git Workflow for Deployment

```bash
# 1. Create feature branch
git checkout -b add-new-tool

# 2. Make changes and commit
git add agent.py
git commit -m "Add weather tool"

# 3. Push to remote
git push origin add-new-tool

# 4. Create pull request (on GitHub/GitLab)

# 5. After review, merge to main
git checkout main
git pull origin main

# 6. Tag release
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# 7. Deploy
docker-compose up -d --build
```

### GitHub Repository Setup

```bash
# Create repository on GitHub, then:
git remote add origin https://github.com/username/agent-repo.git
git branch -M main
git push -u origin main
```

---

## Best Practices

### 1. Configuration Management

Use configuration files:

```python
# config.py
import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = "AI Agent"
    openai_api_key: str
    database_url: str
    redis_url: str
    environment: str = "development"
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### 2. Logging

Implement comprehensive logging:

```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('logs/agent.log', maxBytes=10485760, backupCount=5),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Use in code
logger.info("Agent started")
logger.error(f"Error processing query: {error}")
```

### 3. Error Handling

Graceful error handling:

```python
from fastapi import HTTPException

@app.post("/query")
async def query_agent(query: Query):
    try:
        result = agent.run(query.question)
        return result
    except ValueError as e:
        logger.warning(f"Invalid input: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Internal error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

### 4. Health Checks

Implement health endpoints:

```python
@app.get("/health")
def health_check():
    checks = {
        "api": "ok",
        "database": check_database(),
        "redis": check_redis(),
        "models": check_models_loaded()
    }
    
    if all(v == "ok" for v in checks.values()):
        return {"status": "healthy", "checks": checks}
    else:
        raise HTTPException(status_code=503, detail=checks)
```

### 5. Rate Limiting

Protect your API:

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/query")
@limiter.limit("10/minute")
async def query_agent(request: Request, query: Query):
    return agent.run(query.question)
```

### 6. Monitoring

Add metrics collection:

```python
from prometheus_client import Counter, Histogram, generate_latest

query_counter = Counter('agent_queries_total', 'Total queries')
query_duration = Histogram('agent_query_duration_seconds', 'Query duration')

@app.post("/query")
async def query_agent(query: Query):
    query_counter.inc()
    with query_duration.time():
        result = agent.run(query.question)
    return result

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

---

## Creating a Browser Interface

### Simple HTML Frontend

Create `static/index.html`:

```html
<!DOCTYPE html>
<html>
<head>
    <title>AI Agent</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; }
        #chat { border: 1px solid #ccc; height: 400px; overflow-y: scroll; padding: 10px; }
        .message { margin: 10px 0; }
        .user { color: blue; }
        .agent { color: green; }
        input { width: 80%; padding: 10px; }
        button { padding: 10px 20px; }
    </style>
</head>
<body>
    <h1>AI Agent Chat</h1>
    <div id="chat"></div>
    <input type="text" id="question" placeholder="Ask a question...">
    <button onclick="sendQuery()">Send</button>

    <script>
        async function sendQuery() {
            const question = document.getElementById('question').value;
            if (!question) return;

            // Display user message
            addMessage('user', question);
            document.getElementById('question').value = '';

            // Send to API
            const response = await fetch('/query', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({question: question})
            });

            const data = await response.json();
            addMessage('agent', data.answer);
        }

        function addMessage(role, text) {
            const chat = document.getElementById('chat');
            const msg = document.createElement('div');
            msg.className = `message ${role}`;
            msg.textContent = `${role}: ${text}`;
            chat.appendChild(msg);
            chat.scrollTop = chat.scrollHeight;
        }

        document.getElementById('question').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendQuery();
        });
    </script>
</body>
</html>
```

Serve static files:

```python
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")
```

---

## Saturday Practical Session

### Complete Deployment Project

**Objective:** Deploy your multi-tool agent with browser interface

**Tasks:**

1. **Prepare Application**
   - Create FastAPI wrapper for your agent
   - Add health check endpoint
   - Implement error handling
   - Create simple HTML interface

2. **Containerize**
   - Write Dockerfile
   - Create docker-compose.yml
   - Test locally with Docker

3. **Secure Secrets**
   - Move API keys to .env
   - Add .env to .gitignore
   - Test environment variable loading

4. **Version Control**
   - Initialize Git repository
   - Create .gitignore
   - Make initial commit
   - Push to GitHub

5. **Deploy**
   - Option A: Deploy with Docker Compose locally
   - Option B: Deploy to Google Cloud Run
   - Test deployed endpoint
   - Access via browser

6. **Monitor**
   - Check logs
   - Test health endpoint
   - Verify all tools work
   - Test error handling

**Deliverables:**
- Working agent accessible via browser
- GitHub repository with clean code
- Deployment documentation
- Demo video or screenshots

---

## Requirements File

Create `requirements.txt`:

```
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-dotenv==1.0.0
openai==1.3.0
langchain==0.0.340
chromadb==0.4.18
sentence-transformers==2.2.2
scikit-learn==1.3.2
pandas==2.1.3
requests==2.31.0
pydantic==2.5.0
gunicorn==21.2.0
prometheus-client==0.19.0
slowapi==0.1.9
```

---

## Troubleshooting

### Common Issues

**1. Port Already in Use**
```bash
# Find process using port
lsof -i :8000
# Kill process
kill -9 PID
```

**2. Docker Build Fails**
```bash
# Clear Docker cache
docker system prune -a
# Rebuild without cache
docker-compose build --no-cache
```

**3. Environment Variables Not Loading**
```bash
# Check .env file exists
ls -la .env
# Verify docker-compose loads it
docker-compose config
```

**4. API Key Errors**
```python
# Add validation
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not set")
```

---

## Preparation for Saturday

Before the practical session:

1. **Install Required Tools:**
   ```bash
   # Docker
   # Install from https://docs.docker.com/get-docker/
   
   # Docker Compose (included with Docker Desktop)
   
   # Git
   # Install from https://git-scm.com/downloads
   ```

2. **Test Your Agent:**
   - Ensure all tools work locally
   - Test ML models load correctly
   - Verify RAG system works
   - Check API integrations

3. **Prepare Accounts:**
   - GitHub account
   - (Optional) Google Cloud account with billing enabled

4. **Review Materials:**
   - Docker basics
   - Git commands
   - FastAPI documentation

---

## Conclusion

Congratulations on reaching the final week! By deploying your agent, you're taking the crucial step from development to production. This week's skills enable you to:

- Share your agent with users
- Run agents reliably in production
- Manage secrets securely
- Version and maintain your code
- Scale your agent as needed

You now have the complete skillset to build, deploy, and maintain production AI agents!

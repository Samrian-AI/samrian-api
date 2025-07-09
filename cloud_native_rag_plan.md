# Morphik RAG System: Cloud-Native Deployment and Extraction Guide

## 1. Overview
This document provides a comprehensive guide to extracting and deploying the RAG (Retrieval-Augmented Generation) components from the Morphik system to a cloud-native environment. The architecture is designed for scalability, maintainability, and cost-efficiency while preserving the advanced features of the original implementation.

## 2. Target Architecture

### 2.1 High-Level Architecture

```mermaid
graph TD
    A[Client] -->|HTTP/HTTPS| B[API Gateway]
    B -->|Routes| C[Cloud Run Services]
    
    subgraph Cloud Run Services
        D[Document Ingestion Service]
        E[Query Service]
        F[Embedding Service]
        G[Chunking Service]
    end
    
    C -->|Process Documents| H[Cloud Storage]
    C -->|Store/Query Vectors| I[Vertex AI Matching Engine]
    C -->|Store/Query Metadata| J[Cloud SQL]
    C -->|Process Queries| K[LLM Service]
    
    style D fill:#f9f,stroke:#333
    style E fill:#bbf,stroke:#333
    style F fill:#9cf,stroke:#333
    style G fill:#9fc,stroke:#333
```

### 2.2 Component Breakdown

1. **API Gateway**
   - Routes requests to appropriate services
   - Handles authentication and rate limiting
   - Manages CORS and request validation

2. **Document Ingestion Service**
   - Processes uploaded documents (PDFs, images, text)
   - Extracts text and layout information
   - Generates chunks with metadata
   - Triggers embedding generation

3. **Embedding Service**
   - Generates embeddings using Colpali model
   - Handles batch processing
   - Implements caching for performance

4. **Vector Store**
   - Vertex AI Matching Engine for production
   - Local PGVector for development/testing
   - Handles hybrid search (semantic + metadata)

5. **Metadata Store**
   - Cloud SQL (PostgreSQL)
   - Stores document metadata, chunk information, and relationships
   - Enables efficient filtering and retrieval

6. **LLM Service**
   - Handles prompt construction
   - Manages LLM API calls
   - Implements response streaming

## 3. RAG Component Extraction

### 3.1 Document Processing

#### 3.1.1 Parser Module
- **Location**: `core/parser/`
- **Key Classes**:
  - `MorphikParser`: Main parser for various document types
  - `VideoParser`: Specialized video content processing
  - `StandardChunker`/`ContextualChunker`: Text segmentation

#### 3.1.2 Extraction Process
1. **File Type Detection**: Uses `filetype` library
2. **Content Extraction**: 
   - PDFs: Uses PyMuPDF (fitz) with layout preservation
   - Images: Uses Colpali model for OCR and visual understanding
   - Videos: Extracts frames and processes as images
3. **Chunking**:
   - Splits content into manageable pieces
   - Preserves context across chunks
   - Handles special elements (tables, figures, etc.)

### 3.2 Embedding Generation

#### 3.2.1 Embedding Model
- **Model**: `tsystems/colqwen2.5-3b-multilingual-v1.0`
- **Features**:
  - Multimodal (text + images)
  - Batch processing support
  - Automatic device selection (CPU/GPU/TPU)

#### 3.2.2 Implementation
- **Class**: `ColpaliEmbeddingModel`
- **Key Methods**:
  - `embed_for_ingestion()`: Processes chunks for storage
  - `embed_query()`: Embeds user queries
  - `_process_batch()`: Handles batch processing

### 3.3 Vector Storage

#### 3.3.1 Vector Store Implementation
- **Primary**: `MultiVectorStore`
- **Features**:
  - Hybrid search (semantic + metadata)
  - Support for multiple vector types
  - Efficient similarity search

#### 3.3.2 Schema
```sql
CREATE TABLE multi_vector_embeddings (
    id UUID PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_number INTEGER,
    content TEXT,
    embedding VECTOR(1024),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 3.4 Query Processing

#### 3.4.1 Retrieval
1. **Query Embedding**: Convert query to vector
2. **Similarity Search**: Find relevant chunks
3. **Reranking**: Optional neural reranking

#### 3.4.2 Generation
1. **Prompt Construction**:
   - System prompt
   - Retrieved context
   - User query
   - Conversation history
2. **LLM Integration**:
   - LiteLLM for model abstraction
   - Support for multiple providers (OpenAI, Anthropic, etc.)
   - Streaming responses

## 4. Extraction and Deployment

### 4.1 Code Extraction

#### 4.1.1 Core Components to Extract
1. **From `core/parser/`**:
   - `morphik_parser.py`
   - `video_parser.py`
   - Chunking utilities

2. **From `core/embedding/`**:
   - `colpali_embedding_model.py`
   - Base embedding interfaces

3. **From `core/vector_store/`**:
   - `multi_vector_store.py`
   - Base vector store interfaces

4. **From `core/services/`**:
   - Document processing logic
   - Query handling
   - Integration points

#### 4.1.2 Dependencies
```bash
# Core Dependencies
pillow>=10.0.0  # Image processing
pymupdf>=1.22.0  # PDF processing
colpali-engine>=1.0.0  # Multimodal embeddings
pgvector>=0.2.0  # Vector operations
litellm>=1.0.0  # LLM abstraction

# Optional Dependencies
torch>=2.0.0  # For local embedding generation
transformers>=4.30.0  # For local models
```

### 4.2 Infrastructure Setup

#### 4.2.1 Prerequisites
```bash
gcloud services enable \
  run.googleapis.com \
  aiplatform.googleapis.com \
  sqladmin.googleapis.com \
  storage.googleapis.com \
  vpcaccess.googleapis.com \
  servicenetworking.googleapis.com
```

### 4.3 Infrastructure as Code (Terraform)

#### 4.3.1 Main Configuration
```hcl
# variables.tf
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

# main.tf
provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable required services
resource "google_project_service" "services" {
  for_each = toset([
    "run.googleapis.com",
    "aiplatform.googleapis.com",
    "sqladmin.googleapis.com",
    "storage.googleapis.com",
    "vpcaccess.googleapis.com",
    "servicenetworking.googleapis.com"
  ])
  
  service = each.key
  disable_on_destroy = false
}

# Cloud SQL (PostgreSQL with pgvector)
resource "google_sql_database_instance" "vector_db" {
  name             = "morphik-vector-db"
  database_version = "POSTGRES_14"
  region           = var.region
  
  settings {
    tier = "db-f1-micro"
    
    ip_configuration {
      ipv4_enabled = true
      private_network = google_compute_network.vpc.id
    }
  }
  
  depends_on = [google_service_networking_connection.private_vpc_connection]
}

# Cloud Storage for documents
resource "google_storage_bucket" "documents" {
  name     = "${var.project_id}-documents"
  location = var.region
}

# Vertex AI Index
resource "google_vertex_ai_index" "vector_index" {
  display_name = "morphik-vector-index"
  description  = "Vector index for semantic search"
  
  metadata {
    contents_delta_uri = "gs://${google_storage_bucket.documents.name}/index"
    config {
      dimensions = 1024
      approximate_neighbors_count = 150
      distance_measure_type = "DOT_PRODUCT_DISTANCE"
      algorithm_config {
        tree_ah_config {
          leaf_node_embedding_count = 500
          leaf_nodes_to_search_percent = 7
        }
      }
    }
  }
  
  region = var.region
}
```

### 4.4 Containerization

#### 4.4.1 Dockerfile
```dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Ensure scripts in .local are usable
ENV PATH="/root/.local/bin:${PATH}"

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run as non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
```

#### 4.4.2 docker-compose.yml
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8080:8080"
    environment:
      - ENVIRONMENT=development
      - DATABASE_URL=postgresql://user:pass@db:5432/morphik
      - VECTOR_STORE_TYPE=pgvector
    volumes:
      - .:/app
    depends_on:
      - db

  db:
    image: ankane/pgvector:latest
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=morphik
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

### 4.5 Deployment

#### 4.5.1 Build and Push Container
```bash
# Set environment variables
export PROJECT_ID=your-project-id
export REGION=us-central1

# Build and push container
gcloud builds submit --tag gcr.io/$PROJECT_ID/morphik-rag

# Deploy to Cloud Run
gcloud run deploy morphik-rag \
  --image gcr.io/$PROJECT_ID/morphik-rag \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --set-env-vars="PROJECT_ID=$PROJECT_ID" \
  --set-env-vars="ENVIRONMENT=production" \
  --set-env-vars="VECTOR_INDEX_ID=$(gcloud ai indexes list --region=$REGION --format='value(name)' --filter='displayName=morphik-vector-index')" \
  --cpu=2 \
  --memory=4Gi \
  --min-instances=0 \
  --max-instances=10

# Enable VPC connector for private network access
gcloud run services update morphik-rag \
  --vpc-connector=morphik-vpc-connector \
  --vpc-egress=all \
  --region=$REGION
```

#### 4.5.2 Database Setup
```bash
# Enable pgvector extension
psql $DATABASE_URL -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Create tables and indexes
psql $DATABASE_URL -f sql/schema.sql
```

## 5. Testing

### 5.1 Unit Tests
```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run tests
pytest tests/unit/
```

### 5.2 Integration Tests
```bash
# Start test services
docker-compose -f docker-compose.test.yml up -d

# Run integration tests
pytest tests/integration/
```

### 5.3 Load Testing
```bash
# Install k6
brew install k6

# Run load test
k6 run --vus 10 --duration 30s tests/load/ingestion_test.js
```

## 6. Configuration

### 6.1 Environment Variables

#### 6.1.1 Required Variables
```env
# Application
ENVIRONMENT=production  # or 'development', 'staging'
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR

# GCP
PROJECT_ID=your-project-id
REGION=us-central1

# Storage
STORAGE_BUCKET=your-bucket-name
STORAGE_PROVIDER=gcs  # or 'local' for development

# Vector Store
VECTOR_STORE=vertex_ai  # or 'pgvector' for development
VECTOR_INDEX_ID=projects/[PROJECT_NUMBER]/locations/[REGION]/indexes/[INDEX_ID]

# Database
DATABASE_URL=postgresql://user:pass@host:5432/morphik
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10

# Embeddings
EMBEDDING_MODEL=tsystems/colqwen2.5-3b-multilingual-v1.0
EMBEDDING_DEVICE=cuda  # or 'cpu', 'mps' for M1 Mac
EMBEDDING_BATCH_SIZE=8

# LLM
LLM_PROVIDER=openai  # or 'anthropic', 'vertexai', etc.
LLM_MODEL=gpt-4-1106-preview
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000

# Authentication
AUTH_REQUIRED=true
JWT_SECRET=your-jwt-secret
API_KEYS=key1,key2,key3
```

#### 6.1.2 Optional Variables
```env
# Caching
REDIS_URL=redis://localhost:6379/0
CACHE_TTL=3600  # seconds

# Monitoring
SENTRY_DSN=your-sentry-dsn
OTEL_SERVICE_NAME=morphik-rag
OTEL_EXPORTER_OTLP_ENDPOINT=otel-collector:4317

# Rate Limiting
RATE_LIMIT=100  # requests per minute
RATE_LIMIT_WINDOW=60  # seconds
```

### 6.2 Configuration Management

#### 6.2.1 Config Class
```python
# app/config.py
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    # Application
    environment: str = Field(..., env="ENVIRONMENT")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    # GCP
    project_id: str = Field(..., env="PROJECT_ID")
    region: str = Field("us-central1", env="REGION")
    
    # Vector Store
    vector_store: str = Field("vertex_ai", env="VECTOR_STORE")
    vector_index_id: str = Field(..., env="VECTOR_INDEX_ID")
    
    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()
```

#### 6.2.2 Usage in Application
```python
from app.config import settings

# Access configuration
print(f"Environment: {settings.environment}")
print(f"Project ID: {settings.project_id}")

# Database connection
from sqlalchemy import create_engine
engine = create_engine(settings.database_url)
```

## 7. CI/CD Pipeline

### 7.1 GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy to Cloud Run

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

env:
  PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
  REGION: us-central1
  SERVICE: morphik-rag

jobs:
  build-and-deploy:
    name: Build and Deploy
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
    
    # Setup gcloud CLI
    - name: Set up Cloud SDK
      uses: google-github-actions/setup-gcloud@v1
      with:
        project_id: ${{ secrets.GCP_PROJECT_ID }}
        service_account_key: ${{ secrets.GCP_SA_KEY }}
        export_default_credentials: true
    
    # Configure Docker to use gcloud as credential helper
    - run: gcloud --quiet auth configure-docker
    
    # Build and push container
    - name: Build and Push Container
      run: |
        docker build -t gcr.io/$PROJECT_ID/$SERVICE:$GITHUB_SHA .
        docker push gcr.io/$PROJECT_ID/$SERVICE:$GITHUB_SHA
    
    # Deploy to Cloud Run
    - name: Deploy to Cloud Run
      run: |
        gcloud run deploy $SERVICE \
          --image gcr.io/$PROJECT_ID/$SERVICE:$GITHUB_SHA \
          --platform managed \
          --region $REGION \
          --allow-unauthenticated \
          --update-env-vars ENVIRONMENT=production \
          --update-secrets=DATABASE_URL=projects/$PROJECT_ID/secrets/DB_CONNECTION_STRING:latest \
          --service-account morphik-sa@$PROJECT_ID.iam.gserviceaccount.com
    
    # Run database migrations
    - name: Run Migrations
      run: |
        gcloud run jobs execute migrate-db \
          --region $REGION \
          --wait \
          --update-env-vars ENVIRONMENT=production \
          --update-secrets=DATABASE_URL=projects/$PROJECT_ID/secrets/DB_CONNECTION_STRING:latest
```

### 7.2 Cloud Build Configuration

```yaml
# cloudbuild.yaml
steps:
  # Run tests
  - name: 'python:3.11'
    id: test
    entrypoint: bash
    args:
      - '-c'
      - |
        pip install -r requirements-test.txt
        pytest tests/

  # Build container
  - name: 'gcr.io/cloud-builders/docker'
    id: build
    args:
      - 'build'
      - '--build-arg'
      - 'ENVIRONMENT=production'
      - '-t'
      - 'gcr.io/$PROJECT_ID/$_SERVICE:$_TAG'
      - '-t'
      - 'gcr.io/$PROJECT_ID/$_SERVICE:latest'
      - '.'
    dir: 'services/api'

  # Push container
  - name: 'gcr.io/cloud-builders/docker'
    id: push
    args:
      - 'push'
      - 'gcr.io/$PROJECT_ID/$_SERVICE:$_TAG'

  # Deploy to Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    id: deploy
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - '$_SERVICE'
      - '--image'
      - 'gcr.io/$PROJECT_ID/$_SERVICE:$_TAG'
      - '--platform'
      - 'managed'
      - '--region'
      - '$_REGION'
      - '--allow-unauthenticated'
      - '--update-env-vars'
      - 'ENVIRONMENT=$_ENV'
      - '--revision-suffix'
      - '${SHORT_SHA}'

# Configuration options
substitutions:
  _SERVICE: 'morphik-rag'
  _REGION: 'us-central1'
  _ENV: 'production'

# Tags for the build
tags:
  - 'rag'
  - 'cloud-run'

# Timeout and machine type
timeout: 1800s  # 30 minutes
options:
  machineType: 'E2_HIGHCPU_8'
  logging: CLOUD_LOGGING_ONLY
```

## 8. Observability

### 8. Observability

#### 8.1 Monitoring

##### 8.1.1 Key Metrics
```yaml
# metrics.yaml
metrics:
  - name: requests_total
    type: counter
{{ ... }}
    type: counter
    description: Total tokens processed by LLM
    labels: [type]  # prompt or completion
```

##### 8.1.2 Alerting Rules
```yaml
# alerts.yaml
groups:
- name: api
  rules:
{{ ... }}
      severity: warning
    annotations:
      summary: High latency on {{ $labels.endpoint }}
      description: '99th percentile latency is {{ $value }}s for {{ $labels.endpoint }}'
```

#### 8.2 Logging

##### 8.2.1 Log Structure
```json
{
  "timestamp": "2023-11-15T12:00:00Z",
  "level": "INFO",
  "message": "Request completed",
  "request_id": "abc123",
  "endpoint": "/api/chat",
  "method": "POST",
  "status_code": 200,
  "duration_ms": 245,
  "user_agent": "Mozilla/5.0...",
  "ip": "192.168.1.1",
  "user_id": "user-123",
  "metadata": {
    "model": "gpt-4",
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150
  }
}
```

##### 8.2.2 Logging Configuration
```python
# app/logging.py
import logging
import json
import sys
from typing import Dict, Any

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'message': record.getMessage(),
            'logger': record.name,
        }
        
        # Add extra fields
        if hasattr(record, 'extra'):
            log_record.update(record.extra)
        
        # Handle exceptions
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_record, ensure_ascii=False)

def setup_logging(level=logging.INFO):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)
    
    # Configure third-party loggers
    logging.getLogger('uvicorn').handlers = []
    logging.getLogger('uvicorn').propagate = True
    
    return root_logger
```

#### 8.3 Distributed Tracing

##### 8.3.1 OpenTelemetry Configuration
```python
# app/tracing.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

def setup_tracing(service_name: str):
    # Configure the tracer provider
    provider = TracerProvider()
    processor = BatchSpanProcessor(OTLPSpanExporter())
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    
    # Instrument FastAPI application
    FastAPIInstrumentor().instrument()
    SQLAlchemyInstrumentor().instrument()
    RequestsInstrumentor().instrument()
    
    return trace.get_tracer(service_name)
```

##### 8.3.2 Trace Context Propagation
```python
# app/middleware/trace.py
from fastapi import Request, Response
from opentelemetry import trace
from opentelemetry.trace.status import Status, StatusCode

async def trace_middleware(request: Request, call_next):
    tracer = trace.get_tracer(__name__)
    
    with tracer.start_as_current_span(f"{request.method} {request.url.path}") as span:
        # Add request context
        span.set_attributes({
            "http.method": request.method,
            "http.url": str(request.url),
            "http.host": request.url.hostname,
            "http.scheme": request.url.scheme,
            "http.target": request.url.path,
        })
        
        # Process request
        try:
            response = await call_next(request)
            
            # Add response context
            span.set_attribute("http.status_code", response.status_code)
            span.set_status(Status(StatusCode.OK))
            
            return response
            
        except Exception as e:
            # Record the exception
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
```

## 9. Scaling and Performance

### 9.1 Auto-scaling
- **Min instances**: 0 (to reduce costs)
- **Max instances**: 10 (adjust based on load)
- **CPU**: 2 vCPUs per instance
- **Memory**: 4GB per instance
- **Concurrency**: 80 (adjust based on workload)

### 9.2 Caching
- **Redis**: For session storage and caching
- **CDN**: For static assets and model files

## 10. Security

### 10.1 IAM
- **Principle of least privilege**
- **Service accounts** for each service
- **Secret management** with Secret Manager

### 10.2 Network Security
- **VPC Service Controls**
- **Private IP** for internal services
- **Cloud Armor** for DDoS protection

## 11. Cost Optimization

### 11.1 Resource Management
- **Preemptible VMs** for non-critical workloads
- **Commitment discounts** for long-running resources
- **Autoscaling** to handle variable loads

### 11.2 Monitoring
- **Budget alerts**
- **Cost breakdown** by service
- **Idle resource detection**

## 12. Next Steps

1. **Setup Infrastructure**:
   ```bash
   terraform init
   terraform apply -var="project_id=$PROJECT_ID"
   ```

2. **Deploy Application**:
   ```bash
   gcloud builds submit --config cloudbuild.yaml
   ```

3. **Verify Deployment**:
   ```bash
   gcloud run services describe morphik-rag --region us-central1
   ```

4. **Monitor Performance**:
   - Check Cloud Monitoring dashboards
   - Review Cloud Logging for errors
   - Set up alerts for critical metrics
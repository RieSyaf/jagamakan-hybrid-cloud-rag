# Hybrid Local-Cloud RAG Pipeline ☁️ 🧠

A cost-optimized Retrieval-Augmented Generation (RAG) system that combines AWS Cloud Storage with Local LLM inference. This project demonstrates how to build secure, scalable AI pipelines without incurring high GPU cloud costs.

## 🏗 Architecture


The pipeline follows a "Hybrid" approach:
1.  **Storage (AWS S3):** Raw documents (PDFs) are securely stored in the cloud using `boto3`.
2.  **Ingestion (ETL):** A Python script downloads files via IAM credentials, chunks text using LangChain, and generates embeddings locally.
3.  **Vector Database (PostgreSQL):** Embeddings are stored in a Dockerized Postgres instance using `pgvector`.
4.  **Inference (Ollama):** The Llama 3.2 model runs locally to synthesize answers, keeping inference costs at $0.

## 🛠 Tech Stack
* **Cloud:** AWS S3, AWS IAM
* **AI/LLM:** LangChain, Ollama (Llama 3.2), Nomic Embeddings
* **Database:** PostgreSQL (pgvector), Docker
* **Language:** Python 3.12

## 🚀 Key Features
* **Cost Engineering:** Zero-cost inference by utilizing local hardware for the "thinking" (LLM) and "embedding" steps.
* **Data Security:** Uses `python-dotenv` to manage AWS Access Keys, ensuring no credentials are hardcoded.
* **Hallucination Control:** System prompt constrained to "Answer ONLY from context," preventing the AI from making up facts (tested with out-of-domain queries).

## 💻 How to Run
### 1. Prerequisites
* Docker Desktop installed
* Ollama installed (`ollama pull llama3.2`)
* AWS S3 Bucket with a PDF uploaded

### 2. Setup
```bash
# Clone the repo
git clone [https://github.com/YOUR_USERNAME/hybrid-cloud-rag-pipeline.git](https://github.com/YOUR_USERNAME/hybrid-cloud-rag-pipeline.git)

# Install dependencies
pip install -r requirements.txt

# Start Database
docker run -d --name local-postgres -e POSTGRES_USER=myuser -e POSTGRES_PASSWORD=mypassword -e POSTGRES_DB=vector_db -p 5432:5432 pgvector/pgvector:pg16

### 3. Usage

#Ingest Data: Downloads from S3 and saves to Vector DB.
- run python ingest_s3.py in terminal

#Start Chat:
- run python chat_manual.py in terminal

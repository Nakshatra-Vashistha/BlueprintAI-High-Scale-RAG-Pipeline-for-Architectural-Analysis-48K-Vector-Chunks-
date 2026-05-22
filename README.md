# BlueprintAI-High-Scale-RAG-Pipeline-for-Architectural-Analysis-48K-Vector-Chunks-
An end-to-end, asynchronous RAG pipeline that processes natural language queries to retrieve and synthesize source-cited compliance answers from 48,000+ architectural document chunks. Built on a high-performance FastAPI backend, the system integrates Pinecone vector search, Gemini 2.5 Flash, and Supabase token authentication.

# Architecture Research RAG: Production-Grade Vector Search & Compliance Engine

An end-to-end, asynchronous Retrieval-Augmented Generation (RAG) pipeline designed to ingest, clean, index, and intelligently query massive volumes of dense architectural design manuals, building regulations, and structural compliance codes.

![Python 3.10](https://img.shields.io/badge/Python-3.10-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-v0.104.1-green) ![Pinecone](https://img.shields.io/badge/Pinecone-VectorDB-blueviolet) ![Supabase](https://img.shields.io/badge/Supabase-Auth%20%26%20DB-emerald) ![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-orange)

---

## 📖 Project Description
An end-to-end, asynchronous RAG pipeline that processes natural language queries to retrieve and synthesize source-cited compliance answers from **48,000+ architectural document chunks**. Built on a high-performance **FastAPI** backend, the system integrates **Pinecone** vector search, **Gemini 2.5 Flash**, and **Supabase** token authentication with a decoupled **Next.js** user interface.

---

## 🚀 Key Capabilities
* **48,000+ Production Vector Chunks:** Custom data pipeline optimized to clean, parse, and embed deep architectural text fragments into dense vector spaces.
* **Network-Resilient Ingestion:** Built with deterministic ID generation, chunk cooling-off delays, and safe batching loops to ensure reliable, duplicate-free database updates.
* **Asynchronous Web Architecture:** High-performance backend serving both single-query search tools and stateful, multi-turn interactive chat sessions.

---

## 🏗️ System Tech Stack
* **Vector Database:** Pinecone (Serverless AWS index, Cosine metric mapping)
* **Embedding Model:** `multi-qa-MiniLM-L6-cos-v1` (Optimized for technical Q&A context)
* **LLM Engine:** `gemini-2.5-flash` via the modern `google-genai` SDK
* **Relational DB & Auth:** Supabase Client (Streamlined user authentication and profile tracking)
* **Web Framework:** FastAPI managed with Uvicorn server processes

---

## 🛠️ Quick Start

```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME

# Set up virtual environment & install dependencies
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Configure environment variables (.env)
# GEMINI_API_KEY=... | PINECONE_API_KEY=... | SUPABASE_URL=... | SUPABASE_KEY=...

# Initialize and seed the Vector DB
python main.py --init

# Launch API server
uvicorn api:app --reload --host 0.0.0.0 --port 8000

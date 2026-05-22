import json
import os
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from supabase import create_client, Client

from embedding_utils import EmbeddingModel
from config import config

class ResearchPaperDatabase:
    def __init__(self):
        self.config = config
        
        # ==========================================
        # 1. Initialize Supabase (Relational DB & Auth)
        # ==========================================
        print("\n🔍 [DB] Connecting to Supabase...")
        if config.SUPABASE_URL and config.SUPABASE_KEY:
            # Raw string diagnostics to track unexpected encoding/formatting artifacts
            print("🔍 [RAW STRING DEBUG]")
            print(f"URL: '{config.SUPABASE_URL}'")
            print(f"KEY FIRST 10: '{str(config.SUPABASE_KEY)[:10]}'")
            print(f"KEY LAST 10: '{str(config.SUPABASE_KEY)[-10:]}'")
            print(f"KEY LENGTH: {len(str(config.SUPABASE_KEY))} characters")
            
            try:
                self.supabase: Client | None = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
                print("✅ [DB] Supabase client initialized successfully.")
            except Exception as e:
                print(f"❌ [DB] Supabase client creation failed: {e}")
                raise e
        else:
            print("[WARNING] Supabase credentials missing. Set SUPABASE_URL and SUPABASE_KEY.")
            self.supabase = None
            
        # ==========================================
        # 2. Initialize Pinecone (Vector DB)
        # ==========================================
        print("[DB] Connecting to Pinecone...")
        if not config.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is not set in environment variables.")
            
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index_name = config.PINECONE_INDEX_NAME
        
        # ==========================================
        # 3. Initialize Embedding Model
        # ==========================================
        self.embedding_model = EmbeddingModel(config.EMBEDDING_MODEL)
        
        # ==========================================
        # 4. Connect to or create the Vector Index
        # ==========================================
        self.index = self._get_or_create_index()

    def _get_or_create_index(self):
        """Creates the Pinecone index if it doesn't exist."""
        existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"[DB] Creating new Pinecone index: {self.index_name}...")
            self.pc.create_index(
                name=self.index_name,
                dimension=384, # Matches multi-qa-MiniLM-L6-cos-v1
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        else:
            print(f"[DB] Connected to existing Pinecone index: {self.index_name}")
            
        return self.pc.Index(self.index_name)

    def load_jsonl_file(self, file_path: str):
        """Extract text and format metadata for Pinecone"""
        documents, metadatas, ids = [], [], []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(tqdm(lines, desc=f"Processing {file_path}")):
                    try:
                        data = json.loads(line.strip())
                        content = data.get('text', '')
                        metadata = data.get('metadata', {})

                        if not metadata or not isinstance(metadata, dict):
                            metadata = {"source": os.path.basename(file_path)}

                        if content and len(content.strip()) > 0:
                            documents.append(content)
                            # Pinecone requires strings/numbers in metadata. 
                            # We store the raw text here so we can retrieve it later.
                            metadata['text'] = content
                            metadatas.append(metadata)
                            ids.append(f"{os.path.basename(file_path)}_{i}")

                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {i} in {file_path}: {e}")
                        continue
        except FileNotFoundError:
            print(f"File not found: {file_path}")

        return documents, metadatas, ids

    def add_documents_from_jsonl(self, jsonl_files: list[str]):
        """Embeds text and uploads vectors to Pinecone in batches with retry logic"""
        import time
        from tqdm import tqdm

        all_documents, all_metadatas, all_ids = [], [], []

        # 1. Load and aggregate data from all files
        for file_path in jsonl_files:
            docs, metas, ids = self.load_jsonl_file(file_path)
            all_documents.extend(docs)
            all_metadatas.extend(metas)
            all_ids.extend(ids)

        if not all_documents:
            print("[DB] No documents to add to the database.")
            return

        print(f"[DB] Total documents loaded: {len(all_documents)}")
        print("[DB] Embedding chunks and streaming to Pinecone...")

        # 2. Process chunks, embed them, and upsert with network resilience
        batch_size = 100
        for i in tqdm(range(0, len(all_documents), batch_size), desc="Uploading to Pinecone"):
            batch_docs = all_documents[i:i + batch_size]
            batch_metas = all_metadatas[i:i + batch_size]
            batch_ids = all_ids[i:i + batch_size]

            # Convert text chunks to vector embeddings
            batch_vectors = [self.embedding_model.embed_text(doc) for doc in batch_docs]

            # Zip into (id, vector, metadata) expected by Pinecone
            # Wrapped in list() to satisfy strict Pylance/Type Check requirements
            upsert_data = list(zip(batch_ids, batch_vectors, batch_metas))
            
            # Network resilience loop (up to 3 attempts per batch)
            for attempt in range(3):
                try:
                    self.index.upsert(vectors=upsert_data)
                    break  # Success! Break the retry loop and move to the next batch
                except Exception as e:
                    if attempt == 2:  # Last try failed completely
                        print(f"\n❌ [FATAL] Batch upload failed completely after 3 attempts.")
                        raise e
                    print(f"\n⚠️ [WARNING] Network drop or timeout on batch. Retrying in 3 seconds... ({e})")
                    time.sleep(3)

            # Tiny cooling-off delay to avoid slamming the network socket connection
            time.sleep(0.1)

        print(f"[DB] Successfully added {len(all_documents)} documents to Pinecone.")

    def query_documents(self, query: str, n_results: int = config.TOP_K_RESULTS):
        """Query Pinecone and format output to match the legacy ChromaDB schema"""
        try:
            # 1. Embed user query
            query_vector = self.embedding_model.embed_text(query)

            # 2. Search Pinecone
            response = self.index.query(
                vector=query_vector,
                top_k=n_results,
                include_metadata=True
            )

            # 3. Format output so rag_pipeline.py doesn't break
            docs, metas, distances = [], [], []
            for match in response['matches']:
                docs.append(match['metadata'].get('text', ''))
                metas.append(match['metadata'])
                distances.append(1.0 - match['score']) # Convert Pinecone similarity to distance

            return {
                'documents': [docs],
                'metadatas': [metas],
                'distances': [distances]
            }
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return None

    def get_collection_stats(self):
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_documents": stats.get('total_vector_count', 0),
                "collection_name": self.index_name
            }
        except Exception as e:
            return {"error": str(e)}

    def persist(self):
        # Pinecone is fully managed in the cloud; no local persistence needed.
        pass
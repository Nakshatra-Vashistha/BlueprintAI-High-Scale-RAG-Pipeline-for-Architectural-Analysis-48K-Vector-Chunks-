import os
from pathlib import Path
from dotenv import load_dotenv

# Lock in the base directory first
BASE_DIR = Path(__file__).resolve().parent.parent 

# 1. ADD override=True to smash cached/stale system terminal variables
load_dotenv(BASE_DIR / ".env", override=True)

class Config:
    BASE_DIR = BASE_DIR
    
    # ==========================================
    # Cloud Credentials
    # ==========================================
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = "architecture-rag"
    
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    # ==========================================
    # Embedding & Database Settings
    # ==========================================
    EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"
    
    IS_LOCAL = os.path.exists(r"D:\RAG\Architect-RAG-LLM-Assistant - DEPLOYED VERSION")

    if IS_LOCAL:
        LOCAL_DATA_DIR = Path(r"D:\RAG\Project\DATAset pdfs && cleaned jsonl\chunks")
        JSONL_FILES = [
            str(LOCAL_DATA_DIR / "building_codes_chunks.jsonl"),
            str(LOCAL_DATA_DIR / "case_studies_chunks.jsonl"),
            str(LOCAL_DATA_DIR / "material_guide_chunks.jsonl"),
            str(LOCAL_DATA_DIR / "misc_chunks.jsonl")
        ]
    else:
        JSONL_FILES = [
            str(BASE_DIR / "chunks" / "building_codes_chunks.jsonl"),
            str(BASE_DIR / "chunks" / "case_studies_chunks.jsonl"),
            str(BASE_DIR / "chunks" / "material_guide_chunks.jsonl"),
            str(BASE_DIR / "chunks" / "misc_chunks.jsonl")
        ]
    
    GEMINI_MODEL = "gemini-2.5-flash"  
    TOP_K_RESULTS = 10
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    TRUNCATE_DOC_CHARS = 5000  
    GENERATION_MAX_TOKENS = 2048  

config = Config()

# 2. SANITY CHECK: If it still reads empty/broken, pull the emergency brake immediately
if not config.SUPABASE_KEY or "your-" in config.SUPABASE_KEY.lower():
    print(f"\n❌ [CRITICAL] Config failed to load a valid Supabase Key.")
    print(f"👉 Expected Path: {BASE_DIR / '.env'}")
    print(f"👉 Current value: '{config.SUPABASE_KEY}'\n")
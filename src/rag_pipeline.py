import time
from typing import List, Dict, Any
import google.generativeai as genai
from config import config
from database import ResearchPaperDatabase

class RAGPipeline:
    def __init__(self):
        self.db = ResearchPaperDatabase()
        
        # Initialize Gemini API Client
        try:
            genai.configure(api_key=config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(config.GEMINI_MODEL)
            print(f"[RAG] Successfully initialized Gemini model: {config.GEMINI_MODEL}")
        except Exception as e:
            print(f"[RAG] Failed to initialize Gemini API: {e}")
            print("Make sure you have added a valid GEMINI_API_KEY to config.py")
    
    def generate_response(self, query: str, context: List[str], metadatas: List[Dict[str, Any]]) -> str:
        """Generate response using Gemini API with retrieved context and metadata"""
        
        # Prepare expanded context and include the actual document titles for better citations
        truncated = [doc[:config.TRUNCATE_DOC_CHARS] for doc in context]
        context_text = ""
        for i, doc in enumerate(truncated):
            # FIX: Fallback to 'source' (filename) if 'title' is not explicitly in the metadata
            title = metadatas[i].get('title') or metadatas[i].get('source') or 'Technical Document'
            context_text += f"REFERENCE {i+1} ({title}):\n{doc}\n\n"

        # THE "STRICT & ACCURATE" PROMPT: Forces direct answers and prevents hallucinated fluff
        prompt = f"""
        You are a technical research assistant. Your PRIMARY GOAL is to answer the 
        user's specific question using ONLY the facts found in the technical context.
        
        STRICT GUIDELINES:
        1. DIRECTNESS: Answer the user's question in the very first paragraph.
        2. EVIDENCE: Every claim must be backed by a Source/Reference number and Title.
        3. NO FLUFF: Avoid long background introductions. Jump straight into the technical answer.
        4. ACCURACY: If the provided text does not contain the specific answer, state clearly that the information is not in the research papers. Do not add general knowledge not found in the sources.
        5. STRUCTURE: Use clear headings and bullet points for technical specifications or data.
        
        TECHNICAL CONTEXT:
        {context_text}
        
        USER QUESTION: {query}
        
        TECHNICAL ANSWER:
        """

        try:
            print("[RAG] Sending prompt to Gemini... (this is usually very fast)")
            t0 = time.time()
            
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=config.GENERATION_MAX_TOKENS,
                    temperature=0.7, 
                )
            )
            
            elapsed = time.time() - t0
            print(f"[RAG] Gemini generation completed in {elapsed:.2f}s")
            return response.text

        except Exception as e:
            elapsed = time.time() - t0 if 't0' in locals() else 0
            print(f"[RAG] Gemini call failed after {elapsed:.2f}s: {e}")
            return f"Error generating response: {str(e)}"
    
    def query(self, user_query: str, n_results: int = config.TOP_K_RESULTS) -> Dict[str, Any]:
        """Complete RAG pipeline: retrieve and generate"""
        # Step 1: Query the database
        print("[RAG] Searching for relevant research papers...")
        t0 = time.time()
        
        # Use the wrapper method defined in your database.py
        results = self.db.query_documents(user_query, n_results=n_results)
        
        retrieval_time = time.time() - t0
        print(f"[RAG] Retrieval completed in {retrieval_time:.2f}s")
        
        # Safety check if no results are found
        if not results or not results['documents'] or len(results['documents'][0]) == 0:
            return {
                "answer": "No relevant research papers found for your query.",
                "sources": [],
                "context": []
            }
        
        # Extract retrieved documents
        retrieved_docs = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0] if 'distances' in results else [0] * len(metadatas)
        
        # Step 2: Generate response
        print("[RAG] Generating comprehensive answer...")
        t0 = time.time()
        
        # Passed metadatas to the generation function so Gemini knows the document titles
        answer = self.generate_response(user_query, retrieved_docs, metadatas)
        
        generation_time = time.time() - t0
        print(f"[RAG] Total generation call time: {generation_time:.2f}s")
        
        # Prepare source information for the frontend/API response
        sources = []
        for i, (metadata, distance) in enumerate(zip(metadatas, distances)):
            # FIX: Fallback logic for the frontend source list as well
            display_title = metadata.get('title') or metadata.get('source') or "Architectural Document"
            
            source_info = {
                "source_id": i+1,
                "title": display_title,
                "authors": metadata.get('authors', ["N/A"]),
                "year": metadata.get('year', "N/A"),
                "confidence": f"{1 - distance:.3f}" if distance is not None else "N/A"
            }
            sources.append(source_info)
        
        return {
            "answer": answer,
            "sources": sources,
            "context": retrieved_docs,
            "query": user_query
        }
    
    def initialize_database(self, jsonl_files: List[str]):
        """Initialize the database with research papers"""
        print("Initializing database with research papers...")
        # Matched the expected method name for your database structure
        self.db.add_documents_from_jsonl(jsonl_files)
        
        # If your database class uses a persist method, keep it, otherwise it safely ignores it
        if hasattr(self.db, 'persist'):
            self.db.persist()
            
        print("Database initialization complete!")
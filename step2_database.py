import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the local embedding AI model (lightweight, runs completely on your CPU)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

class NPTELVectorDB:
    def __init__(self, db_path="nptel_study.db"):
        # 1. Open a direct connection to a local database file on your hard drive
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        with self.conn:
            # 2. Schema definition: Store raw text, source files, page numbers, and vector bytes
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_base (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT,
                    source TEXT,
                    page INTEGER,
                    vector BLOB
                )
            """)

    def insert_chunks(self, chunks: list[dict]):
        """Converts text paragraphs into numerical coordinates and stores them."""
        texts = [c["text"] for c in chunks]
        print(f"\n[AI EMBEDDING] Converting text chunks to numbers using SentenceTransformer...")
        
        # 3. Vector Conversion: Translate English text into multidimensional numeric coordinates
        embeddings = embedding_model.encode(texts)

        with self.conn:
            for chunk, vec in zip(chunks, embeddings):
                # Convert the numpy float array list into raw bytes for SQL BLOB storage
                vec_blob = vec.astype(np.float32).tobytes()
                
                # Safe extraction supporting different metadata key formats
                meta = chunk.get("metadata", {})
                source = meta.get("source_file") or meta.get("source", "unknown")
                page = meta.get("page_number") or meta.get("page", 0)

                self.conn.execute(
                    "INSERT INTO knowledge_base (text, source, page, vector) VALUES (?, ?, ?, ?)",
                    (chunk["text"], source, page, vec_blob)
                )
                
        print(f"[DATABASE] Saved {len(chunks)} text chunks to your local database successfully.")

    def query_similarity(self, query_text: str, top_k: int = 4) -> list[dict]:
        """Calculates vectorized Cosine Similarity to find matching slide contexts instantly."""
        # Pass a string directly to get a 1D array of shape (dim,)
        query_vec = embedding_model.encode(query_text).astype(np.float32)

        cursor = self.conn.cursor()
        cursor.execute("SELECT text, source, page, vector FROM knowledge_base")
        rows = cursor.fetchall()

        if not rows:
            return []

        texts, sources, pages, blobs = zip(*rows)
        
        # Reconstruct all numeric coordinates into a 2D NumPy matrix
        db_vectors = np.array([np.frombuffer(b, dtype=np.float32) for b in blobs])

        # Core RAG Math: Vectorized Cosine Similarity (Blazing fast matrix multiplication)
        dot_products = np.dot(db_vectors, query_vec)
        norm_query = np.linalg.norm(query_vec)
        norm_db = np.linalg.norm(db_vectors, axis=1)

        # Prevent division by zero
        safe_norms = norm_db * norm_query
        safe_norms[safe_norms == 0] = 1e-10
        similarities = dot_products / safe_norms

        # Get indices of top_k highest scoring matches
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "text": texts[idx],
                "metadata": {"source": sources[idx], "page": pages[idx]},
                "score": float(similarities[idx])
            })
            
        return results
"""
Agent 3: Semantic Retrieval of Past Incidents
Loads incident data, embeds reason + action_plan into ChromaDB,
and provides semantic search to find similar past incidents.
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Global variables (initialized lazily)
embedding_model = None
chroma_client = None
collection = None
INCIDENTS_PATH = "data/clean/incidents.csv"


def initialize_db():
    """
    Initialize the embedding model and ChromaDB collection.
    This is done lazily to avoid running at import time.
    """
    global embedding_model, chroma_client, collection
    
    if embedding_model is not None:
        return  # Already initialized
    
    # Load incident data
    df = pd.read_csv(INCIDENTS_PATH)
    
    # Handle NaN values by converting to empty strings
    df = df.fillna("")
    
    # Initialize embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Initialize ChromaDB client
    chroma_client = chromadb.Client(Settings(
        anonymized_telemetry=False,
        allow_reset=True
    ))
    
    # Create or get the collection
    collection = chroma_client.get_or_create_collection(
        name="incidents",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Combine reason and action_plan into a single text field
    df["combined_text"] = df["reason"].astype(str) + " " + df["action_plan"].astype(str)
    
    # Generate embeddings for all incidents
    embeddings = embedding_model.encode(
        df["combined_text"].tolist(),
        show_progress_bar=True
    )
    
    # Clear existing data if any (to avoid duplicates on re-runs)
    if collection.count() > 0:
        collection.delete(where={})
    
    # Add documents to ChromaDB
    collection.add(
        ids=df["incident_id"].tolist(),
        embeddings=embeddings.tolist(),
        documents=df["combined_text"].tolist(),
        metadatas=[{"reason": str(r), "action_plan": str(a)} 
                   for r, a in zip(df["reason"], df["action_plan"])]
    )
    
    print(f"Indexed {len(df)} incidents in ChromaDB")


# ── 5. Retrieval function ─────────────────────────────────────────────────────────
def retrieve(query: str, top_k: int = 3):
    """
    Retrieve the top-k most similar past incidents for a given query.
    
    Args:
        query: The incident description or query text to search for
        top_k: Number of similar incidents to return (default: 3)
    
    Returns:
        List of dictionaries containing incident_id, reason, action_plan, 
        and similarity score for each match
    """
    # Initialize the database if not already done
    initialize_db()
    
    # Embed the query using the same model
    query_embedding = embedding_model.encode([query]).tolist()
    
    # Query ChromaDB for similar documents
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    
    # Format results
    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "incident_id": results["ids"][0][i],
            "reason": results["metadatas"][0][i]["reason"],
            "action_plan": results["metadatas"][0][i]["action_plan"],
            "similarity_score": 1 - results["distances"][0][i]  # Convert distance to similarity
        })
    
    return retrieved


# ── Explanation of Cosine Similarity ───────────────────────────────────────────────
"""
Cosine Similarity:
------------------
Cosine similarity measures the cosine of the angle between two vectors in a 
high-dimensional space. It ranges from -1 to 1, where:
- 1 = identical direction (perfectly similar)
- 0 = orthogonal (unrelated)
- -1 = opposite direction (completely dissimilar)

Formula: cos(θ) = (A · B) / (||A|| × ||B||)

Unlike Euclidean distance, cosine similarity is:
- Scale-invariant: document length doesn't affect similarity
- Focuses on semantic direction rather than magnitude
- Ideal for text embeddings where word frequency varies

Why Semantic Search over SQL Matching:
---------------------------------------
1. SQL matching (LIKE, =) requires exact keyword matches or simple patterns.
   It fails when:
   - Users use different terminology (e.g., "fabric defect" vs "material flaw")
   - Typos or variations exist
   - The intent is similar but wording differs

2. Semantic search with embeddings:
   - Captures meaning and context, not just keywords
   - Handles synonyms, paraphrases, and related concepts
   - Works across languages and domain-specific terminology
   - Returns results ranked by relevance, not just presence of keywords

Example:
- Query: "thread breakage during stitching"
- SQL LIKE "%thread%" might miss "yarn breakage" or "sewing thread snapped"
- Semantic search finds all semantically similar incidents regardless of exact wording
"""

# ── Example usage (uncomment to test) ─────────────────────────────────────────────
# if __name__ == "__main__":
#     query = "fabric defect during weaving"
#     results = retrieve(query, top_k=3)
#     for r in results:
#         print(f"ID: {r['incident_id']}")
#         print(f"Reason: {r['reason']}")
#         print(f"Action: {r['action_plan']}")
#         print(f"Similarity: {r['similarity_score']:.3f}\n")

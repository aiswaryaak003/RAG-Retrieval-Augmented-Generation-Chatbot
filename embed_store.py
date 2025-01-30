import faiss
import numpy as np
from data_preprocessing import chunk_mapping, chunk_embeddings, model

# Initialize vector store
vector_dim = 384
index = faiss.IndexFlatL2(vector_dim)

# Add embeddings to the index
def add_to_index(chunk_embeddings):
    vectors = []
    chunk_ids = []
    for chunk_id, embedding in chunk_embeddings:
        vectors.append(embedding.numpy())  # Convert tensor to numpy array
        chunk_ids.append(chunk_id)

    # Convert list of vectors to a numpy array
    vectors = np.vstack(vectors)

    # Add vectors to the FAISS index
    index.add(vectors)

    return chunk_ids

# Retrieve top-k chunks from the index
def retrieve_chunks(query, k=5):
    query_vector = model.encode(query, convert_to_tensor=True).numpy()
    distances, indices = index.search(np.array([query_vector]), k)

    # Map indices back to chunk IDs
    retrieved_chunks = [chunk_mapping[chunk_id] for chunk_id in indices[0] if chunk_id < len(chunk_mapping)]
    return retrieved_chunks

# Add embeddings to the vector store
chunk_ids = add_to_index(chunk_embeddings)
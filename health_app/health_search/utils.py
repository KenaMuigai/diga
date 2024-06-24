import faiss
import numpy as np

def create_index():
    dimension = 300  # Example dimension
    index = faiss.IndexFlatL2(dimension)
    return index

def update_index(index, data):
    vectors = np.array([item['vector'] for item in data]).astype('float32')
    index.add(vectors)

def search_index(index, query_vector, k=5):
    query_vector = np.array([query_vector]).astype('float32')
    distances, indices = index.search(query_vector, k)
    return indices

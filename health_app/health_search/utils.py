import faiss
import numpy as np

def create_index():
    # Adjust the dimension according to the vector size you are using
    dimension = 300  # Example dimension, update it as per your actual vector dimension
    index = faiss.IndexFlatL2(dimension)
    return index

def update_index(index, data):
    # Assuming `data` is a list of dictionaries with 'id' and 'vector' keys
    vectors = np.array([item['vector'] for item in data]).astype('float32')
    index.add(vectors)

def search_index(index, query_vector, k=5):
    query_vector = np.array([query_vector]).astype('float32')
    distances, indices = index.search(query_vector, k)
    return indices

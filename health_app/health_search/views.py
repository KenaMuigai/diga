from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
from .models import HealthApplication
from .utils import create_index, search_index, update_index

def search(request):
    query = request.GET.get('q')
    if query:
        query_vector = generate_vector(query)  # Function to convert query to vector
        index = create_index()
        
        # Assuming we have precomputed vectors for health applications
        data = [
            {"id": app.id, "vector": app.vector} for app in HealthApplication.objects.all()
        ]
        update_index(index, data)
        
        indices = search_index(index, query_vector)
        results = [HealthApplication.objects.get(id=data[i]['id']) for i in indices[0]]
        
        return JsonResponse({"results": [app.name for app in results]})
    return JsonResponse({"results": []})

def generate_vector(text):
    # Function to convert text to vector (e.g., using a pre-trained model like BERT)
    return np.random.rand(300)  # Placeholder for actual vector

# Create your views here.

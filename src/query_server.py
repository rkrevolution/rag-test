from flask import Flask, request, jsonify
import faiss
import numpy as np
import requests
import os

app = Flask(__name__)

# Load the FAISS index and chunks
index = faiss.read_index(os.path.join('data', 'faiss_index.bin'))
with open(os.path.join('data', 'chunks.txt'), 'r', encoding='utf-8') as file:
    chunks = file.readlines()

@app.route('/query', methods=['POST'])
def handle_query():
    user_query = request.json['query']

    # Generate query embedding using LocalAI
    def generate_query_embedding(query):
        url = "http://localhost:8080/v1/embeddings"
        headers = {"Content-Type": "application/json"}
        data = {"input": query}
        response = requests.post(url, headers=headers, json=data)
        return response.json()['data'][0]['embedding']

    query_embedding = generate_query_embedding(user_query)
    query_embedding_np = np.array([query_embedding]).astype('float32')

    # Retrieve relevant documents
    k = 5  # Number of relevant chunks to retrieve
    distances, indices = index.search(query_embedding_np, k)
    retrieved_docs = [chunks[i] for i in indices[0]]

    # Generate response using LocalAI
    def generate_response(retrieved_docs, query):
        url = "http://localhost:8080/v1/completions"
        headers = {"Content-Type": "application/json"}
        context = " ".join(retrieved_docs)
        data = {
            "model": "text-davinci-003",
            "prompt": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:",
            "max_tokens": 150
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()['choices'][0]['text'].strip()

    response = generate_response(retrieved_docs, user_query)
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
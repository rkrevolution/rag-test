from flask import Flask, request, jsonify, render_template
import requests
import faiss
import numpy as np

app = Flask(__name__)

# Load the FAISS index
index = faiss.read_index("data/faiss_index.bin")

# Load the chunks
with open("data/chunks.txt", "r", encoding="utf-8") as f:
    chunks = f.readlines()

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    try:
        if request.content_type == 'application/json':
            data = request.json
            query_text = data['input']
        else:
            query_text = request.form['query']

        # Generate embedding for the query
        response = requests.post("http://localhost:8080/v1/embeddings", json={"input": query_text})
        query_embedding = np.array(response.json()['data'][0]['embedding']).astype('float32').reshape(1, -1)

        # Search the FAISS index
        k = 5  # Number of nearest neighbors to retrieve
        D, I = index.search(query_embedding, k)

        # Get the corresponding chunks
        results = [chunks[i].strip() for i in I[0]]

        if request.content_type == 'application/json':
            return jsonify({"results": results})
        else:
            return render_template('results.html', query=query_text, results=results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
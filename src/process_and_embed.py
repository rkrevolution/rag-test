import os
import faiss
import numpy as np
import requests
import fitz  # PyMuPDF
import time

# Path to the PDF file
PDF_FILE_PATH = os.path.join('data', 'first90.pdf')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Extract text from the PDF
pdf_text = extract_text_from_pdf(PDF_FILE_PATH)

# Split the text into lines
text_data = pdf_text.split('\n')

# Split the text into chunks (e.g., paragraphs)
def chunk_text(text, chunk_size=100, overlap=20):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(" ".join(text[i:i + chunk_size]))
    return chunks

# Set chunk size and overlap
CHUNK_SIZE = 50  # Adjust this value as needed
OVERLAP = 10     # Adjust this value as needed

chunks = chunk_text(text_data, CHUNK_SIZE, OVERLAP)

# Calculate the number of chunks to process (adjustable percentage of total)
total_chunks = len(chunks)
percentage_to_process = 0.01  # Adjust this value to change the percentage
num_chunks_to_process = max(1, int(total_chunks * percentage_to_process))

print(f"Total number of chunks: {total_chunks}")
print(f"Number of chunks to process ({percentage_to_process * 100}%): {num_chunks_to_process}")

# Ask user if they want to proceed
proceed = input(f"Do you want to proceed with processing {percentage_to_process * 100}% of the document? (yes/no): ").strip().lower()

if proceed != 'yes':
    print("Process aborted by the user.")
    exit()

print("Starting chunking...")

# Generate embeddings for chunks using LocalAI
def generate_embeddings(chunks, num_chunks):
    url = "http://localhost:8080/v1/embeddings"
    headers = {"Content-Type": "application/json"}
    embeddings = []
    start_time = time.time()
    last_chunk_time = start_time

    for i in range(num_chunks):
        chunk = chunks[i]
        data = {"input": chunk}
        response = requests.post(url, headers=headers, json=data)
        embedding = response.json()['data'][0]['embedding']
        embeddings.append(embedding)

        # Time indicators
        current_time = time.time()
        elapsed_time = current_time - start_time
        current_chunk_time = current_time - last_chunk_time
        avg_time_per_chunk = elapsed_time / (i + 1)
        remaining_chunks = num_chunks - (i + 1)
        estimated_time_remaining = remaining_chunks * avg_time_per_chunk

        # Clear the line and update progress
        print(f"\rProcessed chunk {i + 1}/{num_chunks} - "
              f"Time elapsed: {elapsed_time:.2f}s - "
              f"Current chunk time: {current_chunk_time:.2f}s - "
              f"Avg time per chunk: {avg_time_per_chunk:.2f}s - "
              f"Estimated time remaining: {estimated_time_remaining:.2f}s", end="", flush=True)

        last_chunk_time = current_time

    print()  # Print a newline after the progress is complete
    return embeddings, avg_time_per_chunk

# Process the selected percentage of the chunks
print("Processing chunks:")
start_time = time.time()
embeddings, avg_time_per_chunk = generate_embeddings(chunks, num_chunks_to_process)
total_time = time.time() - start_time

# Save the average chunk time to src/timerecord.py
with open(os.path.join('src', 'timerecord.py'), 'w') as f:
    f.write(f"AVERAGE_CHUNK_TIME = {avg_time_per_chunk}\n")

# Convert embeddings to numpy array
embedding_dim = len(embeddings[0])
embeddings_np = np.array(embeddings).astype('float32')

# Create a FAISS index
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings_np)

# Save the index and chunks for later use
faiss.write_index(index, os.path.join('data', 'faiss_index.bin'))
with open(os.path.join('data', 'chunks.txt'), 'w', encoding='utf-8') as file:
    for chunk in chunks[:num_chunks_to_process]:
        file.write(chunk + '\n')

print(f"\nEmbeddings generated and stored successfully for {percentage_to_process * 100}% of the document.")
print(f"Total processing time: {total_time:.2f} seconds")
print(f"Average time per chunk: {avg_time_per_chunk:.2f} seconds")
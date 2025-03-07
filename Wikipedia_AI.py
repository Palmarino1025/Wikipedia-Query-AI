import requests
from bs4 import BeautifulSoup
import os
from sentence_transformers import SentenceTransformer
import json
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5ForConditionalGeneration, T5Tokenizer
import numpy as np

# Function to fetch the Wikipedia page
def fetch_wikipedia_page(url, filename):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Ensure we notice bad responses (4xx and 5xx)
        
        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.find('div', {'id': 'bodyContent'})
        
        if content:
            text = '\n'.join([p.get_text() for p in content.find_all('p')])
        else:
            text = "No content found."
        
        save_path = os.path.join(os.getcwd(), filename)
        with open(save_path, 'w', encoding='utf-8') as file:
            file.write(text)
        print(f"File saved successfully at: {save_path}")
    except requests.RequestException as e:
        print(f"Error fetching the page: {e}")
    except FileNotFoundError:
        print("Error: Specified directory does not exist.")

# Function to load and split the document into chunks
def load_and_split_document(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
        print(f"Document split into {len(chunks)} chunks.")
        return chunks
    except FileNotFoundError:
        print("Error: File not found.")
        return []

# Function to generate embeddings for text chunks
def generate_embeddings(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return {chunks[i]: embeddings[i].tolist() for i in range(len(chunks))}

# Function to save embeddings to a JSON file
def save_embeddings(embeddings, filename):
    save_path = os.path.join(os.getcwd(), filename)
    with open(save_path, 'w', encoding='utf-8') as file:
        json.dump(embeddings, file)
    print(f"Embeddings saved successfully at: {save_path}")

# Function to retrieve relevant text chunks based on cosine similarity
def retrieve_relevant_chunks(query, embeddings_dict):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query])
    
    embeddings = np.array([embedding for embedding in embeddings_dict.values()])
    texts = list(embeddings_dict.keys())
    
    similarities = cosine_similarity(query_embedding, embeddings)
    
    top_indices = similarities.argsort()[0][-3:][::-1]
    
    top_chunks = [texts[i] for i in top_indices]
    return top_chunks

# Function to generate a response using the HuggingFace model
def generate_response(query, top_chunks):
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    
    prompt = " ".join(top_chunks) + " Question: " + query
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    
    outputs = model.generate(**inputs, max_length=200)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# Main execution
if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    document_filename = "Selected_Document.txt"
    embeddings_filename = "Embeddings.json"
    
    # Fetch and process the Wikipedia page
    fetch_wikipedia_page(url, document_filename)
    chunks = load_and_split_document(document_filename)
    embeddings = generate_embeddings(chunks)
    save_embeddings(embeddings, embeddings_filename)
    
    # Prompt the user for a query
    query = input("Enter your query: ")
    
    # Retrieve the top 3 relevant chunks based on the query
    relevant_chunks = retrieve_relevant_chunks(query, embeddings)
    
    # Generate a response based on the relevant chunks
    response = generate_response(query, relevant_chunks)
    
    print("\nGenerated Response:")
    print(response)

# Optional: Generate requirements.txt and install libraries
""" def install_libraries():
    os.system("pip install sentence-transformers transformers scikit-learn sentencepiece torch torchvision torchaudio")

# Generate requirements.txt
requirements_content = 
requests
beautifulsoup4
sentence-transformers
transformers
scikit-learn
sentencepiece
torch
torchvision
torchaudio
"""

"""with open("requirements.txt", "w") as req_file:
    req_file.write(requirements_content)

print("requirements.txt file created.") """

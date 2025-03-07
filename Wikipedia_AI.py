import requests
from bs4 import BeautifulSoup
import os
from sentence_transformers import SentenceTransformer
import json

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

def load_and_split_document(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        chunks = text.split('\n\n')  # Split by double newlines
        return chunks
    except FileNotFoundError:
        print("Error: File not found.")
        return []

def generate_embeddings(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return {chunks[i]: embeddings[i].tolist() for i in range(len(chunks))}

def save_embeddings(embeddings, filename):
    save_path = os.path.join(os.getcwd(), filename)
    with open(save_path, 'w', encoding='utf-8') as file:
        json.dump(embeddings, file)
    print(f"Embeddings saved successfully at: {save_path}")

if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    document_filename = "Selected_Document.txt"
    embeddings_filename = "Embeddings.json"
    
    fetch_wikipedia_page(url, document_filename)
    chunks = load_and_split_document(document_filename)
    embeddings = generate_embeddings(chunks)
    save_embeddings(embeddings, embeddings_filename)

# Installation commands
def install_libraries():
    os.system("pip install sentence-transformers transformers scikit-learn torch torchvision torchaudio")

# Generate requirements.txt
requirements_content = """requests
beautifulsoup4
sentence-transformers
transformers
scikit-learn
torch
torchvision
torchaudio
"""

with open("requirements.txt", "w") as req_file:
    req_file.write(requirements_content)

print("requirements.txt file created.")

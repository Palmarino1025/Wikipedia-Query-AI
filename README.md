
# **Prompts**
## **Selecting and Loading Document**
Write Python code to read this website https://en.wikipedia.org/wiki/Artificial_intelligence . If the document is a web page, use BeautifulSoup to extract the relevant text. I need to save this in a text file called Selected_Document.txt

## **Installing Required Libraries**
I need the linux install command for all the needed requirements and i need them outlined in a requirements.txt

## **Embeddings**
Write Python code that reads the content of a file called "Selected_Document.txt" and splits it into chunks separated by double newline characters ("\n\n"). Make it use SentenceTransformers library and the model "all-MiniLM-L6-v2" to generate embeddings for a list of text chunks. Store the embeddings in a Python dictionary along with their associated text.

## **Querying the System**
Write Python code that takes a query as input, generates its embedding using SentenceTransformers, and finds the top 3 most similar text chunks from a dictionary of embeddings using cosine similarity from scikit-learn.
Write Python code that uses the HuggingFace Transformers library and the "google/flan-t5-small" model to generate a text response based on retrieved text chunks and a query. Combine the chunks into a single prompt before generating the response. The program should take user queries as input rather than hardcoding them.

I used the two prompts one after the other

## **Deepening my Understanding**
What are five important questions I should ask to understand how this program works? Please include questions about concepts such as cosine similarity, sentence-transformers, and the role of embeddings.

---

# **Reflection Report**

## **Name and Explanation of the Document**
The document used is the Wikipedia article on **Artificial Intelligence**. This article overviewed the different parts of understand artifical intelligence

---
## **How the Program Works**
I asked the AI to give me a simple explanation of how the program works. It responded:

### **1. Load the Document**
- The program starts by loading the document selected by the user. 
- If the document is a webpage, the program scrapes the content using BeautifulSoup. 
- The text is cleaned and saved into a file named `Selected_Document.txt`.

### **2. Process and Split the Document**
- The document is read from `Selected_Document.txt` and split into smaller, manageable chunks. 
- These chunks are separated by double newline characters (`\n\n`) to ensure each chunk represents a logical segment of the text.

### **3. Generate Embeddings**
- Using the `SentenceTransformers` library, each text chunk is converted into a numerical representation called an **embedding**. 
- These embeddings are stored in a dictionary, where each chunk is linked to its corresponding vector representation. 
- Embeddings allow the program to calculate the similarity between text segments efficiently.

### **4. Query the System**
- When the user enters a query, it is also converted into an embedding using the same `SentenceTransformers` model. 
- The program compares this query embedding to all the document embeddings using **cosine similarity**, which measures how closely two vectors are aligned. 
- The top three most similar chunks are retrieved.

### **5. Generate a Response**
- The retrieved chunks are combined into a single prompt and passed to a HuggingFace model (e.g., `flan-t5-small`). 
- This model generates a response that combines the user query and the context provided by the retrieved chunks. 
- The response is displayed as the final output.

### **6. Test and Analyze**
- The program can be tested with multiple queries, and its output is analyzed for relevance and accuracy. 

---

## **Questions About the Program**
Below are five questions I asked an AI tool to deepen my understanding of the program, along with their answers:

1. **What are embeddings, and why are they used in this program?**  
   - **Answer:** In this program, embeddings are essential for transforming text into a format that can be efficiently processed by machine learning algorithms. They enable the program to perform semantic comparisons between the query and document chunks and to generate a meaningful response by using the relevant information.

2. **How does the SentenceTransformer model generate embeddings for text, and what is its role in this program?**  
   - **Answer:** The SentenceTransformer model is responsible for converting text into embeddings (dense vector representations), which capture the meaning of the text. These embeddings are used to measure the similarity between the query and the document chunks, allowing the program to retrieve the most relevant information for generating a meaningful response.

3. **What is cosine similarity, and how is it used to retrieve the most relevant text chunks for a given query?**  
   - **Answer:** Cosine similarity is a metric that measures the similarity between two vectors based on their direction. It is used in this program to compare the embeddings of a user query and document chunks, helping to identify which chunks of text are most relevant to the query. The chunks with the highest cosine similarity are retrieved, and their content is used to generate a response to the user's question.

4. **What are the key differences between classification and regression in machine learning, and why is this program using a text generation model instead of one for classification or regression?**  
   - **Answer:** Classification is used for predicting categorical labels, and regression is used for predicting continuous numerical values. These methods are suitable for tasks where the output is discrete or numerical.

Text generation (like using a model such as T5) is used in this program because the goal is to generate human-readable, contextually relevant responses to user queries. Text generation models are capable of producing diverse, coherent, and semantically rich outputs, making them ideal for tasks such as question answering, summarization, or content creation.

5. **How does the program handle different queries and select the most relevant information to generate a response using the HuggingFace model?**  
   - **Answer:** 
   -Query is embedded into a vector.
  -Cosine similarity is calculated between the query and document chunks to identify relevant information.
  -The most similar chunks are retrieved.
  -The HuggingFace T5 model generates a response based on these chunks.
  -A final textual response is provided to the user, answering their query.

This process enables the program to answer various types of queries by retrieving and generating information from a source document in a way that is both contextually relevant and linguistically coherent.

---

## **Performance Analysis**

### **Retrieval Quality**  
The system effectively retrieved a small part and I think once it finds one response it sends that instead of skimming all the data

### **Response Quality**  
The response is small and not really giving my information. 

### **Possible Improvements**  
1. **Researching more into how the retrieval is being done:** Making sure that the whole document is being looked at and not just sending the first answer it finds. 

2. **Building on multiple queries and exit conditions:** Allowing a small conversation to happen to help understand the responses more clearly if more information is needed.

3. **Adding multiple ways of search past the document :** Adding an ability to reach out past the document to find more information if a clarification is needed. 

---

## **Example Queries and Outputs**

### **Query 1:**  
**"What is the history of AI development?"**    
- **Generated Response:**  
  the market for AI had reached over a billion dollars

### **Query 2:**  
**"What are neural networks?"**  
- **Generated Response:**  
  based on a collection of nodes also known as artificial neurons

### **Query 3:**  
**"Can you explain reinforcement learning?"**  
- **Generated Response:**  
  inverse reinforcement learning

---



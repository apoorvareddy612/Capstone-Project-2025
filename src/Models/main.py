#%%
from rank_bm25 import BM25Okapi
import numpy as np
import pandas as pd

# Load Dataset
df = pd.read_csv('/Users/apoorvareddy/Downloads/Academic/DATS6501/data/data.csv') # Ensure your dataset has "title", "text", "source"


import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Function to clean and segment a query
def clean_and_segment_query(query):
    # Lowercase & Remove special characters
    query = re.sub(r"[^a-zA-Z0-9\s]", "", query.lower())
    
    # Tokenize and remove stopwords
    words = [word for word in query.split() if word not in stop_words]

    # Identify distinct topics (using commas or logical separation)
    topics = []
    temp_topic = []
    
    for word in words:
        if word in ["hoax", "fake", "fraud"]:  # Words that indicate topic shift
            if temp_topic:
                topics.append(" ".join(temp_topic))
                temp_topic = []
        temp_topic.append(word)
    
    if temp_topic:
        topics.append(" ".join(temp_topic))

    return topics

# Example Query
query = "Piltdown man hoax, hoax vaccines autism hoax please help me"

# Get cleaned and segmented query
segmented_queries = clean_and_segment_query(query)
print("Segmented Queries:", segmented_queries)


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode all documents
doc_embeddings = model.encode(df["text"].astype(str).tolist(), convert_to_tensor=True)

# Function to Get Relevant Docs Using BERT Embeddings
def get_relevant_docs_bert(queries, doc_embeddings, df, top_k=3):
    relevant_docs = []
    for query in queries:
        query_embedding = model.encode([query], convert_to_tensor=True)
        similarities = cosine_similarity(query_embedding.cpu().numpy(), doc_embeddings.cpu().numpy()).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        relevant_docs.append(list(top_indices))
    return relevant_docs


for sub_query in segmented_queries:
    relevant_docs = get_relevant_docs_bert([sub_query], doc_embeddings, df, top_k=3) 
    # Print Results
    print(f"Query: {sub_query}")
    print(df.iloc[relevant_docs[0][0]]["title"])
    print(f"Relevant Docs: {relevant_docs}\n")

# %%
print(df.iloc[relevant_docs[0][0]]["title"])
# %%
print(relevant_docs)
# %%

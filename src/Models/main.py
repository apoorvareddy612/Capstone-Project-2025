#%%
import spacy
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load SpaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Load Dataset
df = pd.read_csv('/Users/apoorvareddy/Downloads/Academic/DATS6501/data/data.csv')  # Ensure dataset has "title", "text", "source"
print(df.head())
# Load BERT model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode all documents
doc_embeddings = model.encode(df["text"].astype(str).tolist(), convert_to_tensor=True)

# Function to clean and extract key topics
def extract_key_topics(query):
    """
    Extracts meaningful topics from a conversational query using NLP techniques.
    """
    # Lowercase & Remove special characters
    query = re.sub(r"[^a-zA-Z0-9\s,]", "", query.lower())

    # Parse with SpaCy
    doc = nlp(query)

    # Remove unnecessary words (stopwords, auxiliary verbs, etc.)
    keywords = [token.text for token in doc if not token.is_stop and token.pos_ not in ["AUX", "DET", "PRON"]]

    # Named Entity Recognition (NER) to capture key terms
    entities = [ent.text for ent in doc.ents]  

    # Combine extracted keywords and entities
    refined_query = list(set(keywords + entities))

    return refined_query

# Example Query
query = "I don't remember which episode it speaks about Piltdown man hoax please help me"

# Get cleaned and segmented query
segmented_queries = extract_key_topics(query)
print("🔹 Extracted Key Topics:", segmented_queries)

# Function to Get Relevant Docs Using BERT Embeddings
def get_relevant_docs_bert(queries, doc_embeddings, df, top_k=5):
    relevant_docs = []
    for query in queries:
        query_embedding = model.encode([" ".join(query)], convert_to_tensor=True)
        similarities = cosine_similarity(query_embedding.cpu().numpy(), doc_embeddings.cpu().numpy()).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        relevant_docs.append(list(top_indices))
    return relevant_docs

# Fetch relevant documents for each segmented query
# for sub_query in segmented_queries:
relevant_docs = get_relevant_docs_bert([segmented_queries], doc_embeddings, df, top_k=3)
print(f"\n🔹 Query: {segmented_queries}")
print(f"📌 Top Document: {df.iloc[relevant_docs[0][0]]['title']}")
print(f"📜 Relevant Docs: {relevant_docs}")

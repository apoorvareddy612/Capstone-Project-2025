#%%
import spacy
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions

# Downloads
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


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
def extract_keywords(query):
    """
    Extracts key topics from a conversational query using SpaCy and NLTK.
    """
    # Clean text
    query = contractions.fix(query)  # Expand contractions like "don't"
    query = re.sub(r"[^a-zA-Z0-9\s]", "", query.lower())  # Remove punctuation

    # Step 1: NLTK Lemmatization & Stopword Removal
    tokens = nltk.word_tokenize(query)
    filtered = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]

    # Step 2: Use SpaCy for POS tagging and NER
    spacy_doc = nlp(" ".join(filtered))

    # Extract noun chunks, entities, and content-bearing tokens
    noun_chunks = [chunk.text for chunk in spacy_doc.noun_chunks if len(chunk.text.split()) <= 4]
    entities = [ent.text for ent in spacy_doc.ents]
    keywords = [token.text for token in spacy_doc if token.pos_ in ["NOUN", "PROPN", "ADJ"] and not token.is_stop]

    # Combine and deduplicate
    all_keywords = list(set(keywords + entities + noun_chunks))
    return all_keywords

# Example Query
query = "I don't remember which episode it speaks about Piltdown man hoax please help me"

# Get cleaned and segmented query
segmented_queries = extract_keywords(query)
print("🔹 Extracted Key Topics:", segmented_queries)

# Function to Get Relevant Docs Using BERT Embeddings
def get_relevant_docs_bert(queries, doc_embeddings, top_k=5):
    relevant_docs = []
    for query in queries:
        query_embedding = model.encode([" ".join(query)], convert_to_tensor=True)
        similarities = cosine_similarity(query_embedding.cpu().numpy(), doc_embeddings.cpu().numpy()).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        relevant_docs.append(list(top_indices))
    return relevant_docs

# Fetch relevant documents for each segmented query
# for sub_query in segmented_queries:
relevant_docs = get_relevant_docs_bert([segmented_queries], doc_embeddings, top_k=5)
print(f"\n🔹 Query: {segmented_queries}")
print(f"📌 Top Document: {df.iloc[relevant_docs[0][0]]['title']}")
print(f"📜 Relevant Docs: {relevant_docs}")

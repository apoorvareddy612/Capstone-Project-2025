#%%
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
#%%
df = df = pd.read_csv('/Users/apoorvareddy/Downloads/Academic/DATS6501/data/data.csv')
#%%

def preprocess_text(text, for_bm25=True):
    """
    Preprocess text by:
    - Lowercasing
    - Removing special characters
    - Tokenizing
    - Removing stopwords (for BM25)
    - Lemmatizing (optional)
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    words = text.split()

    if for_bm25:
        words = [word for word in words if word not in stop_words]  # Remove stopwords
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatization

    return " ".join(words)

# Apply preprocessing
df["text"] = df["text"].apply(lambda x: preprocess_text(x))
# %%
from rank_bm25 import BM25Okapi
def tokenize_and_filter(text):
    return [word for word in text.lower().split() if word not in stop_words]

# Tokenize transcripts
tokenized_transcripts = [tokenize_and_filter(text) for text in df["text"]]
bm25 = BM25Okapi(tokenized_transcripts)
# %%
def bm25_search(query, top_k=5):
    tokenized_query = tokenize_and_filter(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    for idx in top_indices: 
        print(df.iloc[idx]["title"]+' by '+ df.iloc[idx]["source"])

# %%
# Example query
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def clean_query(query):
    # Tokenize the query
    tokens = nltk.word_tokenize(query)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # Remove punctuation
    filtered_tokens = [word for word in filtered_tokens if word not in string.punctuation]
    
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    # Join the cleaned tokens back into a string
    cleaned_query = " ".join(lemmatized_tokens)
    
    return cleaned_query

# Test with the user's example
query = "I don't remember which episode it speaks about Piltdown man hoax, hoax vaccines autism hoax please help me"
cleaned_query = clean_query(query)
print(bm25_search(cleaned_query))
# %%
from sentence_transformers import SentenceTransformer

# Load SBERT model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Encode all podcast transcripts
document_embeddings = embedding_model.encode(df["text"].tolist(), convert_to_numpy=True)

# %%
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def embedding_search(query, top_k=5):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, document_embeddings).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    for idx in top_indices: 
        print(df.iloc[idx]["title"]+' by '+ df.iloc[idx]["source"])

embedding_search(query)

# %%
def hybrid_search(query, top_k=5):
    tokenized_query = tokenize_and_filter(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, document_embeddings).flatten()

    # Weighted scoring (60% BM25 + 40% embeddings)
    final_scores = [(bm25_scores[i] * 0.6 + similarities[i] * 0.4) for i in range(len(bm25_scores))]
    sorted_indices = np.argsort(final_scores)[-top_k:][::-1]
    
    for idx in sorted_indices: 
        print(df.iloc[idx]["title"]+' by '+ df.iloc[idx]["source"])

hybrid_search(cleaned_query)
# %%
#Multi Vector Querying
import re

def split_into_sentences(text):
    return re.split(r'(?<=[.!?]) +', text)  # Split by punctuation

sentence_embeddings = []
paragraph_embeddings = []

for text in df["text"]:
    sentences = split_into_sentences(text)
    paragraphs = text.split("\n\n")  # Split by paragraphs
    
    sentence_embeddings.append(embedding_model.encode(sentences, convert_to_numpy=True))
    paragraph_embeddings.append(embedding_model.encode(paragraphs, convert_to_numpy=True))

document_embeddings = embedding_model.encode(df["text"].tolist(), convert_to_numpy=True)

# %%
def multi_vector_search(query, top_k=5):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)

    # Compute similarity scores for sentence, paragraph, and document levels
    sentence_scores = np.array([max(cosine_similarity(query_embedding, s).flatten()) for s in sentence_embeddings])
    paragraph_scores = np.array([max(cosine_similarity(query_embedding, p).flatten()) for p in paragraph_embeddings])
    document_scores = cosine_similarity(query_embedding, document_embeddings).flatten()

    # Weighted scoring (Sentence: 30%, Paragraph: 30%, Document: 40%)
    final_scores = 0.4 * sentence_scores + 0.3 * paragraph_scores + 0.3 * document_scores
    sorted_indices = np.argsort(final_scores)[-top_k:][::-1]

    return [df.iloc[idx]["title"] for idx in sorted_indices]

print(multi_vector_search(query))

# %%
#Universal Sentence Encoder
import tensorflow as tf
import tensorflow_hub as hub

# Load the Universal Sentence Encoder from TensorFlow Hub
use_model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
use_model = hub.load(use_model_url)
 
# %%
import numpy as np

# Example dataset: Replace with your actual dataset
documents = df["text"].tolist()  # List of document texts (transcripts)

# Function to generate embeddings using USE
def generate_use_embeddings(texts, model):
    embeddings = model(texts)  # Apply USE model to the list of documents
    return embeddings.numpy()

# Get embeddings for all documents in the dataset
doc_embeddings = generate_use_embeddings(documents, use_model)

# %%
# Generate query embedding
query_embedding = generate_use_embeddings([query], use_model)

# %%
from sklearn.metrics.pairwise import cosine_similarity

# Calculate cosine similarity between query embedding and document embeddings
similarity_scores = cosine_similarity(query_embedding, doc_embeddings)

# Get the top 5 most similar documents (highest cosine similarity scores)
top_k = 5
top_k_indices = similarity_scores[0].argsort()[-top_k:][::-1]

# Display top matching documents
print("\nTop Matching Documents:")
for idx in top_k_indices:
    print(f"Title: {df.iloc[idx]['title']}, Source: {df.iloc[idx]['source']}, Similarity Score: {similarity_scores[0][idx]}")

# %%
#RoBERTa
import torch
from transformers import RobertaTokenizer, RobertaModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import contractions
# Text Preprocessing Function
def clean_text(text):
    #lowercase
    text = text.lower()
    #remove whitespaces
    text = text.strip()
    #text contraction
    text = contractions.fix(text)
    #remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    #remove html tags
    text = re.sub(r'<.*?>', '', text)
    #remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Lemmatize the tokens
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]

    # Join the lemmas into a single string
    lemmatized_text = ' '.join(lemmas)

    return lemmatized_text


df['text'] = df['text'].apply(clean_text)
df['text'] = df['text'].apply(lemmatize_text)

# Load pre-trained RoBERTa model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")

def get_roberta_embedding(text):
    """Convert text into RoBERTa embedding"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use the [CLS] token embedding (first token) as sentence representation
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# Compute embeddings for all documents
df["embedding"] = df["text"].apply(get_roberta_embedding)
#%%
def search_documents(query, top_k=10):
    """Find the most relevant documents based on RoBERTa similarity"""
    query_embedding = get_roberta_embedding(query)

    # Extract document embeddings
    document_embeddings = np.vstack(df["embedding"].values)
    
    # Compute cosine similarity
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]

    # Get top-k results
    top_k_indices = np.argsort(similarities)[::-1][:top_k]

    # Display results
    print("Top Matching Documents:")
    for idx in top_k_indices:
        print(idx)
        print(f"{df.iloc[idx]['title']} by {df.iloc[idx]['source']} (Score: {similarities[idx]:.4f})")
# %%
query = "Sokal"
search_documents(query)
# %%

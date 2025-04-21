import pandas as pd
import numpy as np 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import contractions
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
from sentence_transformers import SentenceTransformer, CrossEncoder
import spacy
import faiss
import torch
from rank_bm25 import BM25Okapi
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
import contractions

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words("english"))
nlp = spacy.load("en_core_web_sm")
#%%
df = pd.read_csv('./data/data.csv')

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

def preprocess_title(title):
    return title.lower().strip()  # Lowercased for embedding but stored as-is

#%%
#clean and lemmatize text
df['text'] = df['text'].apply(clean_text)
df['text'] = df['text'].apply(lemmatize_text)
titles = df['title'].tolist()
original_titles = titles.copy()  # Preserve original format

# Preprocess for embedding
df['title'] = [preprocess_title(title) for title in titles]

# %%
# # Example query
# def clean_query(query):
#     # Tokenize the query
#     tokens = nltk.word_tokenize(query)
    
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    
#     # Remove punctuation
#     filtered_tokens = [word for word in filtered_tokens if word not in string.punctuation]
    
#     # Lemmatize the tokens
#     lemmatizer = WordNetLemmatizer()
#     lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
#     # Join the cleaned tokens back into a string
#     cleaned_query = " ".join(lemmatized_tokens)
    
#     return cleaned_query

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
    return " ".join(all_keywords)

# # Test with the user's example
# query = "I don't remember which episode it speaks about Piltdown man hoax, hoax vaccines autism hoax please help me"
# cleaned_query = clean_query(query)

df["combined_text"] = df["title"] + " " + df["text"]
# %%
#TF-IDF based search
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(df["combined_text"])
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
# %%
#Implement the search function
# Function to search based on query
def tfidf_search(query, top_k=5):
    query_vector = tfidf_vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_k_indices = cosine_similarities.argsort()[-top_k:][::-1]
    return list(top_k_indices)
# %%
#BM-25 based search
def tokenize_and_filter(text):
    return [word for word in text.lower().split() if word not in stop_words]

tokenized_transcripts = [tokenize_and_filter(text) for text in df["combined_text"]]
bm25 = BM25Okapi(tokenized_transcripts)

#%%
#Implement BM25 Search Function
def bm25_search(query, top_k=5):
    tokenized_query = tokenize_and_filter(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    top_k_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
    return list(top_k_indices)

#%%
# Function to perform BM25 search
def bm25_search1(query, top_k=5):
    bm25 = BM25Okapi(tokenized_transcripts)
    # Tokenize and process the query
    tokenized_query = tokenize_and_filter(query)
    # Get BM25 scores for the query
    bm25_scores = bm25.get_scores(tokenized_query)
    # Get top-K documents from BM25
    top_k_indices = np.argsort(bm25_scores)[::-1][:top_k]
    return top_k_indices, bm25_scores

# Function to perform TF-IDF search
def tfidf_search1(query, top_k_bm25_indices):
    # Transform the user query to TF-IDF vector
    query_tfidf = tfidf_vectorizer.transform([query])
    # Compute cosine similarity between the query and BM25 selected documents
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix[top_k_bm25_indices]).flatten()
    return cosine_similarities, tfidf_matrix

# Function to combine BM25 and TF-IDF results
def combined_search(query, top_k=5):
    top_k_bm25_indices, bm25_scores = bm25_search1(query, top_k)
    cosine_similarities, _ = tfidf_search1(query, top_k_bm25_indices)
    bm25_tfidf_scores = [(bm25_scores[idx], cosine_similarities[i]) for i, idx in enumerate(top_k_bm25_indices)]
    final_scores = [(bm25_score * 0.6 + tfidf_score * 0.4) for bm25_score, tfidf_score in bm25_tfidf_scores]
    sorted_indices = np.argsort(final_scores)[::-1]
    sorted_top_k_bm25_indices = [top_k_bm25_indices[i] for i in sorted_indices]
    return sorted_top_k_bm25_indices 

#%%
# Load model and precomputed embeddings
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
document_embeddings = np.load("./src/Models/SentBERT/document_embeddings.npy")
faiss_index = faiss.read_index("./src/Models/SentBERT/faiss_document_index.index")

# # Updated embedding_search using FAISS
# def embedding_search(query, top_k=5):
#     # Embed the query
#     query_embedding = embedding_model.encode([query], convert_to_numpy=True)

#     if query_embedding.ndim == 1:
#             query_embedding = query_embedding.reshape(1, -1)

#     if query_embedding.shape[1] != faiss_index.d:
#         raise ValueError(f"❌ Dimension mismatch: Query has {query_embedding.shape[1]}, but index expects {faiss_index.d}")


#     # Use FAISS for fast similarity search
#     _, top_indices = faiss_index.search(query_embedding, top_k)

#     # Flatten indices to a simple list
#     return top_indices.flatten().tolist()

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
    
    return sorted_indices

#%%
def retrieve_and_rerank(query, top_k=5):
    """ Retrieve relevant documents using SentenceTransformer + FAISS and Re-Rank using Cross-Encoder """
    
    # Load the retriever model (Bi-Encoder)
    retriever = SentenceTransformer("./src/Models/encoder_model/retriever_model", device="cpu")
    
    # Load the re-ranker model (Cross-Encoder)
    reranker = CrossEncoder("./src/Models/encoder_model/reranker_model", device="cpu")
    
    # Load the FAISS index
    faiss_index = faiss.read_index("./src/Models/encoder_model/faiss_index.bin")
    
    
    # Load the corpus (or it can be passed as an argument to avoid re-reading)
    # df = pd.read_csv('/Users/apoorvareddy/Downloads/Academic/DATS6501/data/data.csv')
    # df["combined_text"] = df["title"] + " " + df["text"]
    corpus = df["combined_text"].tolist()

    # Encode the query into the same vector space
    query_embedding = retriever.encode(query, convert_to_tensor=True).cpu().numpy().reshape(1, -1)

    # FAISS Search
    _, retrieved_indices = faiss_index.search(query_embedding.reshape(1, -1), top_k)
    retrieved_indices = retrieved_indices[0]

    # Retrieve the documents based on indices
    retrieved_docs = [corpus[idx] for idx in retrieved_indices]
    query_doc_pairs = [[query, doc] for doc in retrieved_docs]
    
    # Re-Rank using Cross-Encoder
    relevance_scores = reranker.predict(query_doc_pairs)
    
    # Sort results by relevance score (descending order)
    reranked_results = sorted(zip(retrieved_indices, retrieved_docs, relevance_scores), key=lambda x: x[2], reverse=True)

    # Return the sorted list of indices
    return [idx for idx, _, _ in reranked_results]
#%%
# Load tokenizer and base BERT model
tokenizer = BertTokenizer.from_pretrained("./src/Models/bert/bert_saved_model")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()

# Define Self-Attention layer
class SelfAttention(nn.Module):
    def __init__(self, embedding_dim=768, heads=1):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=heads, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output.mean(dim=1)  # mean pooling

# Initialize Self-Attention layer and load trained weights
self_attention = SelfAttention(embedding_dim=768)
self_attention.load_state_dict(torch.load('./src/Models/bert/self_attention.pth', map_location=torch.device('cpu')))
self_attention.eval()

# Search Function using FAISS
def search_bert(query, top_k=5):
    # Load FAISS inside the function
    try:
        index = faiss.read_index("./src/Models/bert/bert_index.index")
    except Exception as e:
        print(f"Failed to load FAISS index: {e}")
        return []

    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    token_embeddings = outputs.last_hidden_state.squeeze(0)
    refined_embedding = self_attention(token_embeddings.unsqueeze(0))
    query_embedding = refined_embedding.squeeze(0).detach().numpy().reshape(1, -1)

    _, top_indices = index.search(query_embedding, top_k)
    return top_indices.flatten().tolist()

#%%
# Load tokenizer and model
roberta_model = RobertaModel.from_pretrained("./src/Models/RoberTa/roberta_saved_model")
tokenizer = RobertaTokenizer.from_pretrained("./src/Models/RoberTa/roberta_saved_model")
roberta_model.eval()

# Load saved embeddings and FAISS index
roberta_embeddings = np.load('./src/Models/RoberTa/roberta_document_embeddings.npy')
faiss_index = faiss.read_index("./src/Models/RoberTa/roberta_index.faiss")

# Define search function
def search_roberta(query, top_k=5):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = roberta_model(**inputs)
    query_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

    # Perform FAISS search
    _, top_indices = faiss_index.search(query_embedding.reshape(1, -1), top_k)
    return top_indices.flatten().tolist()

#%%
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
#%%
df = df = pd.read_csv('/Users/apoorvareddy/Downloads/Academic/DATS6501/data/data.csv')

#%%
#BERT + attention layer
# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()  # Set model to evaluation mode

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, heads=1):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=heads, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output.mean(dim=1)  # Mean pooling after attention
    
# Initialize Self-Attention Layer (same embedding size as BERT)
self_attention = SelfAttention(embedding_dim=768)

def get_bert_embedding(text):
    """Generate refined BERT embeddings with self-attention"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = bert_model(**inputs)

    # Extract token embeddings
    token_embeddings = outputs.last_hidden_state.squeeze(0)

    # Apply Self-Attention
    refined_embedding = self_attention(token_embeddings.unsqueeze(0))  # Add batch dimension

    # Normalize embeddings
    return refined_embedding.squeeze(0).numpy() / np.linalg.norm(refined_embedding.numpy())
#%%
# Compute embeddings
df["embedding"] = df["text"].apply(get_bert_embedding)

def search_documents(query, top_k=5):
    """Find the most relevant documents using BERT + Self-Attention"""
    query_embedding = get_bert_embedding(query)

    # Extract document embeddings
    document_embeddings = np.vstack(df["embedding"].values)

    # Compute cosine similarity
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]

    # Get top-k results
    top_k_indices = np.argsort(similarities)[::-1][:top_k]

    # Display results
    print("Top Matching Documents:")
    for idx in top_k_indices:
        print(f"{df.iloc[idx]['title']} by {df.iloc[idx]['source']} (Score: {similarities[idx]:.4f})")

#%%
#BERT + Cross Attention
class CrossAttention(nn.Module):
    def __init__(self, embedding_dim, heads=1):
        super(CrossAttention, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=heads, batch_first=True)

    def forward(self, query_embedding, document_embedding):
        """
        Applies cross-attention between the query and document.
        """
        attn_output, _ = self.cross_attention(query_embedding, document_embedding, document_embedding)
        return attn_output.mean(dim=1)  # Mean pooling after attention

# Initialize Cross-Attention Layer
cross_attention = CrossAttention(embedding_dim=768)

def get_bert_embedding(text):
    """Generate refined BERT embeddings."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = bert_model(**inputs)

    # Extract token embeddings
    token_embeddings = outputs.last_hidden_state.squeeze(0)

    # Normalize embeddings
    return token_embeddings.unsqueeze(0)  # Add batch dimension

# Compute document embeddings
df["embedding"] = df["text"].apply(get_bert_embedding)
#%%
def search_documents(query, top_k=3):
    """Find the most relevant documents using Cross-Attention."""
    
    # Get query embedding
    query_embedding = get_bert_embedding(query)

    # Compute cross-attention for each document
    similarities = []
    for doc_embedding in df["embedding"]:
        refined_doc_embedding = cross_attention(query_embedding, doc_embedding)
        similarity_score = cosine_similarity(query_embedding.squeeze(0).numpy(), refined_doc_embedding.squeeze(0).numpy())[0][0]
        similarities.append(similarity_score)

    # Get top-k results
    top_k_indices = np.argsort(similarities)[::-1][:top_k]

    # Display results
    print("\nTop Matching Documents:")
    for idx in top_k_indices:
        print(f"{df.iloc[idx]['title']} by {df.iloc[idx]['source']} (Score: {similarities[idx]:.4f})")

#%%

from transformers import DistilBertTokenizer, DistilBertModel


# Load DistilBERT Tokenizer & Model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
base_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Custom Self-Attention Layer
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        return attn_output

# Search Model with Attention
class SearchModel(nn.Module):
    def __init__(self, base_model):
        super(SearchModel, self).__init__()
        self.base_model = base_model
        self.attention = SelfAttention(hidden_dim=768)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state  # (batch_size, seq_length, hidden_dim)
        
        # Apply self-attention
        embeddings = embeddings.permute(1, 0, 2)  # (seq_length, batch_size, hidden_dim)
        attended_embeddings = self.attention(embeddings)  
        attended_embeddings = attended_embeddings.permute(1, 0, 2)  # Back to (batch_size, seq_length, hidden_dim)
        
        return torch.mean(attended_embeddings, dim=1)  # Get sentence-level embedding

# Initialize Search Model
search_model = SearchModel(base_model)
search_model.eval()
#%%
# Convert Text to Embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        embedding = search_model(inputs["input_ids"], inputs["attention_mask"])
    return embedding.squeeze().numpy()

# Generate Document Embeddings
document_embeddings = np.array([get_embedding(text) for text in df["text"]])

# Search Function
def search(query, top_k=3):
    query_embedding = get_embedding(query).reshape(1, -1)
    
    # Compute Similarity
    similarities = cosine_similarity(query_embedding, document_embeddings)[0]
    
    # Get Top-K Results
    top_k_indices = similarities.argsort()[::-1][:top_k]
    
    print(" Top Matching Documents:")
    for idx in top_k_indices:
        print(f"{df.iloc[idx]['title']} ({df.iloc[idx]['source']}) - Score: {similarities[idx]:.4f}")

# Example Query
query_text = "How is AI improving medical science?"
search(query_text)
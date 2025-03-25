
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
    return refined_embedding.squeeze(0).detach().numpy() / np.linalg.norm(refined_embedding.detach().numpy())
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
import torch

# Save self-attention model
torch.save(self_attention.state_dict(), "self_attention.pth")

tokenizer.save_pretrained("bert_saved_model")

# Convert embeddings to NumPy array
document_embeddings = np.vstack(df["embedding"].values)

# Save embeddings and dataset
np.save("bert_document_embeddings.npy", document_embeddings)
df.to_csv("bert_processed_data.csv", index=False)
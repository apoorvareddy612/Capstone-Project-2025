#%%
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
from bertopic import BERTopic
from transformers import RobertaModel, RobertaTokenizer
import torch.nn as nn
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

#%%
def mean_reciprocal_rank(results, relevant_docs):
    """
    Compute MRR (Mean Reciprocal Rank).
    :param results: List of ranked lists of document indices returned by models.
    :param relevant_docs: List of sets containing relevant document indices for each query.
    :return: MRR score
    """
    reciprocal_ranks = []
    for i, ranked_list in enumerate(results):
        for rank, doc_id in enumerate(ranked_list, start=1):
            if doc_id in relevant_docs[i]:
                reciprocal_ranks.append(1 / rank)
                break
        else:
            reciprocal_ranks.append(0)  # No relevant document found
    
    return np.mean(reciprocal_ranks)

def top_n_accuracy(results, relevant_docs, n=5):
    """
    Compute Top-N Accuracy: Whether at least one relevant document appears in top N results.
    """
    correct_predictions = 0
    for i, ranked_list in enumerate(results):
        if any(doc in ranked_list[:n] for doc in relevant_docs[i]):
            correct_predictions += 1
    return correct_predictions / len(results)

def precision_at_k(results, relevant_docs, k=5):
    """
    Compute Precision@K: How many of the top-K retrieved documents are relevant.
    """
    precision_scores = []
    for i, ranked_list in enumerate(results):
        retrieved_top_k = set(ranked_list[:k])
        relevant_set = set(relevant_docs[i])
        precision = len(retrieved_top_k & relevant_set) / k
        precision_scores.append(precision)
    return np.mean(precision_scores)

def average_precision(ranked_list, relevant_set):
    """
    Compute Average Precision (AP) for a single query.
    """
    num_relevant = 0
    precision_sum = 0
    for i, doc in enumerate(ranked_list):
        if doc in relevant_set:
            num_relevant += 1
            precision_sum += num_relevant / (i + 1)
    return precision_sum / len(relevant_set) if relevant_set else 0

def mean_average_precision(results, relevant_docs):
    """
    Compute Mean Average Precision (MAP): Average of Average Precision (AP) across all queries.
    """
    ap_scores = [average_precision(results[i], set(relevant_docs[i])) for i in range(len(results))]
    return np.mean(ap_scores)

def dcg_at_k(ranked_list, relevant_set, k=5):
    """
    Compute Discounted Cumulative Gain (DCG) at K.
    """
    dcg = 0
    print(f"Ranked List: {ranked_list}")
    for i in range(k):
        if ranked_list[i] in relevant_set:
            print(f"Ranked List: {ranked_list[i]} is in relevant set")
            dcg += 1 / np.log2(i + 2)  # Log2 starts at 2 to avoid division by zero
    return dcg

def ndcg_at_k(results, relevant_docs, k=5):
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) at K.
    """
    ndcg_scores = []
    for i, ranked_list in enumerate(results):
        relevant_set = set(relevant_docs[i])
        k = len(relevant_set)
        dcg = dcg_at_k(ranked_list, relevant_set, k)
        ideal_dcg = dcg_at_k(sorted(relevant_set, reverse=True), relevant_set, k)  # Ideal DCG assumes perfect ranking
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
        ndcg_scores.append(ndcg)
    return np.mean(ndcg_scores)

def evaluate_models(models, queries, relevant_docs, top_k=5):
    """
    Evaluate multiple models and return the top 3 based on NDCG and MAP.
    """
    
    model_scores = {}
    print(models)
    for model_name, model_function in models.items():
        all_results = [model_function(query) for query in queries]

        # Handle NoneType results by replacing them with an empty list
        all_results = [res if res is not None else [] for res in all_results]
        print(all_results)

        # Compute all evaluation metrics
        top_n = top_n_accuracy(all_results, relevant_docs, top_k)
        prec_k = precision_at_k(all_results, relevant_docs, top_k)
        map_score = mean_average_precision(all_results, relevant_docs)
        mpr = mean_reciprocal_rank(all_results, relevant_docs)
        ndcg = ndcg_at_k(all_results, relevant_docs, top_k)

        model_scores[model_name] = (top_n, prec_k, map_score, mpr, ndcg)
    print(model_scores)
    # Rank models based on NDCG + MAP (weighted sum for ranking)
    ranked_models = sorted(model_scores.items(), key=lambda x: (x[1][4] + x[1][2]) / 2, reverse=True)

    return ranked_models[:3], model_scores  # Return top 3 models and full scores
#%%

# BERT + Self-Attention
tokenizer = BertTokenizer.from_pretrained("/Users/apoorvareddy/Downloads/Academic/DATS6501/bert/bert_saved_model")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()  # Set model to evaluation mode

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, heads=1):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=heads, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output.mean(dim=1)  

self_attention = SelfAttention(embedding_dim=768)  # Initialize the SelfAttention model with the right dimensions
self_attention.load_state_dict(torch.load('/Users/apoorvareddy/Downloads/Academic/DATS6501/bert/self_attention.pth', map_location=torch.device('cpu')))  # Load the state dictionary to CPU
self_attention.eval()

# Ensure that the device is set to 'cpu' to avoid CUDA issues
device = torch.device('cpu')

# Try loading the BERT embeddings and handle potential errors
try:
    bert_embeddings = np.load('/Users/apoorvareddy/Downloads/Academic/DATS6501/bert/bert_document_embeddings.npy')
    dimension = bert_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  
    index.add(bert_embeddings)  
    faiss.write_index(index, "bert_index.index")
    print("✅ FAISS index saved as 'bert_index.index'")

except FileNotFoundError:
    print("Error: BERT embeddings file not found.")
    bert_embeddings = None

# RoBERTa
roberta_model = RobertaModel.from_pretrained("/Users/apoorvareddy/Downloads/Academic/DATS6501/RoberTa/roberta_saved_model")
tokenizer = RobertaTokenizer.from_pretrained("/Users/apoorvareddy/Downloads/Academic/DATS6501/RoberTa/roberta_saved_model")
roberta_model.eval()  # Set model to evaluation mode
roberta_embeddings = np.load('/Users/apoorvareddy/Downloads/Academic/DATS6501/RoberTa/roberta_document_embeddings.npy')
dimension = roberta_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  
index.add(roberta_embeddings)  
faiss.write_index(index, "roberta_index.faiss")
print("✅ FAISS index saved as 'roberta_index.faiss'")
# roberta_processed_data = pd.read_csv('path_to/processed_data.csv')

# Define paths
save_path = "/Users/apoorvareddy/Downloads/Academic/DATS6501/encoder_model/retrieval_model"
embedding_path = "/Users/apoorvareddy/Downloads/Academic/DATS6501/encoder_model/corpus_embeddings.npy"
faiss_index_path = "/Users/apoorvareddy/Downloads/Academic/DATS6501/encoder_model/faiss_index.bin"

# Search function for BERT + Self-Attention
def search_bert(query, top_k=5):
    """ Find relevant documents using BERT + Self-Attention """
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    token_embeddings = outputs.last_hidden_state.squeeze(0)
    refined_embedding = self_attention(token_embeddings.unsqueeze(0))
    query_embedding = refined_embedding.squeeze(0).detach().numpy()

    similarities = cosine_similarity([query_embedding], bert_embeddings).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]
    return list(top_indices)

# Search function for RoBERTa
def search_roberta(query, top_k=5):
    """ Search for relevant documents using RoBERTa """
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = roberta_model(**inputs)
    query_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

    similarities = cosine_similarity([query_embedding], roberta_embeddings).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]
    return list(top_indices)


def retrieve_and_rerank(query, top_k=5):
    """ Retrieve relevant documents using SentenceTransformer + FAISS and Re-Rank using Cross-Encoder """
    
    # Load the retriever model (Bi-Encoder)
    retriever = SentenceTransformer("/Users/apoorvareddy/Downloads/Academic/DATS6501/encoder_model/retriever_model", device="cpu")
    
    # Load the re-ranker model (Cross-Encoder)
    reranker = CrossEncoder("/Users/apoorvareddy/Downloads/Academic/DATS6501/encoder_model/reranker_model", device="cpu")
    
    # Load the FAISS index
    faiss_index = faiss.read_index("/Users/apoorvareddy/Downloads/Academic/DATS6501/encoder_model/faiss_index.bin")
    
    
    # Load the corpus (or it can be passed as an argument to avoid re-reading)
    df = pd.read_csv('/Users/apoorvareddy/Downloads/Academic/DATS6501/data/data.csv')
    df["combined_text"] = df["title"] + " " + df["text"]
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
# Example: Models dictionary (replace with actual function calls)
models = {
    "RoBERTa": search_roberta,
    "Self-Attention Transformer": search_bert,
    "Bi-Encoder,Cross-Encoder" : retrieve_and_rerank
}

# Example Queries
queries = ["I don't remember which episode it speaks about Piltdown man hoax please help me"]

# Example Relevant Docs (Manually defined for evaluation)
relevant_docs = [[514, 1191, 542, 1342, 722]] 

# Get Top 3 Models and Full Scores
top_models, full_scores = evaluate_models(models, queries, relevant_docs, top_k=5)

# Print Results
print("Top 3 Models based on NDCG & MAP:")
for rank, (model, scores) in enumerate(top_models, start=1):
    print(f"{rank}. {model} -> Top-N: {scores[0]:.4f}, Precision@K: {scores[1]:.4f}, MAP: {scores[2]:.4f}, MPR: {scores[3]:.4f}, NDCG: {scores[4]:.4f}")

print("\nFull Evaluation Scores:")
for model, scores in full_scores.items():
    print(f"{model} -> Top-N: {scores[0]:.4f}, Precision@K: {scores[1]:.4f}, MAP: {scores[2]:.4f}, MPR: {scores[3]:.4f}, NDCG: {scores[4]:.4f}")
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
    for i in range(k):
        if ranked_list[i] in relevant_set:
            dcg += 1 / np.log2(i + 2)  # Log2 starts at 2 to avoid division by zero
    return dcg

def ndcg_at_k(results, relevant_docs, k=5):
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) at K.
    """
    ndcg_scores = []
    for i, ranked_list in enumerate(results):
        relevant_set = set(relevant_docs[i])
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

    for model_name, model_function in models.items():
        all_results = [model_function(query) for query in queries]

        # Compute all evaluation metrics
        top_n = top_n_accuracy(all_results, relevant_docs, top_k)
        prec_k = precision_at_k(all_results, relevant_docs, top_k)
        map_score = mean_average_precision(all_results, relevant_docs)
        ndcg = ndcg_at_k(all_results, relevant_docs, top_k)

        model_scores[model_name] = (top_n, prec_k, map_score, ndcg)

    # Rank models based on NDCG + MAP (weighted sum for ranking)
    ranked_models = sorted(model_scores.items(), key=lambda x: (x[1][3] + x[1][2]) / 2, reverse=True)

    return ranked_models[:3], model_scores  # Return top 3 models and full scores

# Example: Models dictionary (replace with actual function calls)
models = {
    "BM25": bm25_search, 
    "TF-IDF": tfidf_search,  
    "RoBERTa": roberta_search,
    "DistilBERT + Attention": distilbert_attention_search,
    "Cross-Attention Transformer": cross_attention_search,
    "LDA Topic Modeling": lda_search,
    "BERTopic": bertopic_search
}

# Example Queries
queries = ["AI in healthcare", "Machine learning in finance", "Deep learning applications"]

# Example Relevant Docs (Manually defined for evaluation)
relevant_docs = [[1, 2], [0, 3], [2, 4]]

# Get Top 3 Models and Full Scores
top_models, full_scores = evaluate_models(models, queries, relevant_docs, top_k=5)

# Print Results
print("Top 3 Models based on NDCG & MAP:")
for rank, (model, scores) in enumerate(top_models, start=1):
    print(f"{rank}. {model} -> Top-N: {scores[0]:.4f}, Precision@K: {scores[1]:.4f}, MAP: {scores[2]:.4f}, NDCG: {scores[3]:.4f}")

print("\nFull Evaluation Scores:")
for model, scores in full_scores.items():
    print(f"{model} -> Top-N: {scores[0]:.4f}, Precision@K: {scores[1]:.4f}, MAP: {scores[2]:.4f}, NDCG: {scores[3]:.4f}")
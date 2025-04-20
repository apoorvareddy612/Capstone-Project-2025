# Capstone-Project-2025
Capstone project under Professor Yuxiao Huang

## Problem Statement
In today's digital landscape, podcasts have exploded in popularity, leading to vast repositories of audio content. However, efficient information retrieval from podcast transcripts remains a significant challenge. Most platforms rely on keyword-based search, which fails to understand the semantic intent behind user queries—especially when queries are conversational, vague, or contain multiple topics.

For instance, a user might search "I don't remember which episode talks about the Piltdown man hoax and vaccines." A standard keyword search engine might fail to retrieve relevant results due to its inability to understand context, synonyms (e.g., “fraud” vs. “hoax”), or semantic relations. Moreover, users often express their queries naturally, not as precisely engineered keyword lists.

This disconnect between user language and retrieval techniques results in irrelevant matches, poor user satisfaction, and lost content discoverability, especially in long-form content like podcasts.

## Proposed Solution
This project presents a podcast search engine that combines classical and modern NLP retrieval models to handle both keyword-based and semantic queries. It supports several retrieval strategies, giving users the flexibility to search using the most suitable approach for their needs:

Retrieval Models Implemented:
**BM25**: Classic bag-of-words based scoring using term frequency and document frequency.

**TF‑IDF**: Vector space model scoring documents based on term importance.

**BM25 + TF‑IDF (Hybrid Lexical)**: Weighted ensemble that balances precision from BM25 and coverage from TF‑IDF.

**BERT + BM25 (Hybrid Semantic)**: Combines contextual embeddings from Sentence-BERT with BM25 to balance semantics and exact term match.

**BERT + Self-Attention:** A custom-built transformer variant that adds a multi-head attention layer on top of BERT token embeddings to improve representation.

**RoBERTa:** Dense retrieval using the roberta-base transformer for contextual embeddings.

**Encoder-Based Retrieve & Re-Rank:** A two-stage pipeline:

**Stage 1:** Fast retrieval using a Bi-Encoder.

**Stage 2:** Reranking candidates using a Cross-Encoder for precise relevance scoring.

This ensemble of models ensures that the search engine supports both exact-match and semantic-match scenarios, catering to different types of users and queries. Whether the user knows the exact phrase or just remembers the theme, the engine can find the best-matching content.

## Tech Stack
**Languages:** Python

**Libraries:**

`SentenceTransformers`, `transformers`, `faiss` for vector search

`scikit-learn` for TF-IDF and cosine similarity

`rank_bm25` for probabilistic BM25

`nltk`, `spaCy`, contractions for preprocessing

`Streamlit` for the web-based interface

**Models:**

`all-MiniLM-L6-v2`, `roberta-base`, custom BERT + Self-Attention

`cross-encoder/ms-marco-MiniLM-L-6-v2` for reranking

Precomputed FAISS indexes and NumPy embeddings for fast retrieval

## Launching the demo
1. Clone the Repository
```bash
git clone https://github.com/yourusername/podcast-semantic-search.git
cd podcast-semantic-search
```
2. Set Up Virtual Environment
```bash
python3.10 -m venv env
source env/bin/activate  # or use env\Scripts\activate on Windows
pip install -r requirements.txt
```
3. Launch the Streamlit App
```bash
streamlit run app.py
```

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc
import torch

# Garbage collection
gc.collect()

# Clear GPU cache if you're using CUDA (optional, safe on CPU too)
if torch.cuda.is_available():
    torch.cuda.empty_cache()

import streamlit as st
import pandas as pd
import torch
from Models.models import (
    extract_keywords, bm25_search, tfidf_search, combined_search, hybrid_search, retrieve_and_rerank, search_roberta, search_bert
)
#from Models.external import search_bert_via_script



def main():
    # --------------- UI -----------------
    st.set_page_config(page_title="Semantic Search", page_icon="🔍", layout="wide")
    # --------------- DATA ---------------
    @st.cache_data
    def load_df():
        return pd.read_csv("Capstone-Project-2025/data/data.csv")

    df = load_df()

    # --------------- MODEL MAP ----------
    MODEL_FUNCS = {
        "BM25": bm25_search,
        "TF‑IDF": tfidf_search,
        "BM25 + TF‑IDF": combined_search,
        #"SentBERT": embedding_search,
        "BERT + BM25": hybrid_search,
        "BERT + Self-Attention": search_bert,
        "RoBERTa": search_roberta,
        "Encoder": retrieve_and_rerank,
    }



    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox("Choose page", ["Introduction"] + list(MODEL_FUNCS.keys()))
        if page != "Introduction":
            top_k = st.slider("Top‑K results", 3, 20, 5)

    # --------------- INTRO PAGE ---------
    if page == "Introduction":
        st.title("Podmatch: Semantic Search for Podcasts")
        st.image("Capstone-Project-2025/assets/image.png", width=700)
        st.markdown(
            """
            Welcome to Podmatch! This app allows you to search through a curated collection of podcast episodes using various semantic search models.
            You can choose from different search algorithms in the sidebar to find relevant episodes based on your queries.
            """
        )
        st.info("Select a model in the sidebar to start searching.")

    # --------------- MODEL PAGES --------
    else:
        st.title(f"Model: {page}")
        query = st.text_input("Enter your query and press Enter", key="query")

        if query:
            with st.spinner("Searching..."):
                query = extract_keywords(query)  # Clean and segment the query
                indices = MODEL_FUNCS[page](query, top_k=top_k)
            st.subheader(f"{top_k} Similar Results to your query")
            if indices is None or len(indices)==0:
                st.warning("No results returned by this model.")
            else:
                # Accept int‑indices or pre‑formatted strings
                if isinstance(indices[0], int):
                    titles = [df.iloc[i]['title'] for i in indices]
                    sources = [df.iloc[i]['source'] for i in indices]
                else:
                    titles = [df.iloc[i]['title'] for i in indices]
                    sources = [df.iloc[i]['source'] for i in indices]

            for i, (title, source) in enumerate(zip(titles, sources), start=1):
                st.write(f"**{i}. {title} - {source}**")

if __name__ == "__main__":
    main()
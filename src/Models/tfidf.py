#%%
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

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

#%%
df = pd.read_csv('/Users/apoorvareddy/Downloads/Academic/DATS6501/data/data.csv')

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
# Example query
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

# %%
#TF-IDF based search
#Pre-process the data
# Combine title and transcript for better context
df["combined_text"] = df["title"] + " " + df["text"]
# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
# Fit and transform the combined_text
tfidf_matrix = tfidf_vectorizer.fit_transform(df["combined_text"])
# Get the feature names (words) in TF-IDF vector
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
# %%
#Implement the search function
# Function to search based on query
def tfidf_search(query, top_k=5):
    # Transform the user query into TF-IDF vector
    query_vector = tfidf_vectorizer.transform([query])

    # Compute cosine similarity between query and documents
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Get top-k results based on similarity scores
    top_k_indices = cosine_similarities.argsort()[-top_k:][::-1]

    # Format and display results
    print("Top Matching Podcasts:")
    # Display top k results with the original titles
    for idx in top_k_indices:
        original_title = original_titles[idx]  # Get the original title
        print(f"{original_title} by {df.iloc[idx]['source']}")
        
    return list(top_k_indices)

tfidf_search(cleaned_query)
# %%
#BM-25 based search
from rank_bm25 import BM25Okapi
import nltk
# Combine title and transcript for better context
df["combined_text"] = df["title"] + " " + df["text"]
# Ensure NLTK stopwords are downloaded
nltk.download("stopwords")
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

# Tokenize and remove stopwords
def tokenize_and_filter(text):
    return [word for word in text.lower().split() if word not in stop_words]

# Prepare data (tokenized)
tokenized_transcripts = [tokenize_and_filter(text) for text in df["combined_text"]]

# Initialize BM25 model
bm25 = BM25Okapi(tokenized_transcripts)

#%%
#Implement BM25 Search Function
def bm25_search(query, top_k=5):
    # Tokenize and process the query
    tokenized_query = tokenize_and_filter(query)
    
    # Get BM25 scores for the query
    bm25_scores = bm25.get_scores(tokenized_query)

    # Get top-k documents
    top_k_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]

    # Format and display results
    print("Top Matching Podcasts:")
    # After filtering or ranking results, use original_titles to print the unmodified titles
    for idx in top_k_indices:
        original_title = original_titles[idx]  # Get the original title (no need for indexing again)
        print(f"{original_title} by {df.iloc[idx]['source']}")
    return list(top_k_indices)
# Example query
bm25_search(cleaned_query)


# %%
#BM25 + TFIDF based search
from rank_bm25 import BM25Okapi
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# Download NLTK stopwords
nltk.download("stopwords")
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

# Function to tokenize and filter text
def tokenize_and_filter(text):
    return [word for word in text.lower().split() if word not in stop_words]

# Function to perform BM25 search
def bm25_search1(query, top_k=5, corpus=None):
    tokenized_transcripts = [tokenize_and_filter(text) for text in corpus]
    bm25 = BM25Okapi(tokenized_transcripts)
    # Tokenize and process the query
    tokenized_query = tokenize_and_filter(query)
    # Get BM25 scores for the query
    bm25_scores = bm25.get_scores(tokenized_query)
    # Get top-K documents from BM25
    top_k_indices = np.argsort(bm25_scores)[::-1][:top_k]
    return top_k_indices, bm25_scores

# Function to perform TF-IDF search
def tfidf_search1(query, top_k_bm25_indices, corpus=None):
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    # Transform the user query to TF-IDF vector
    query_tfidf = tfidf_vectorizer.transform([query])
    # Compute cosine similarity between the query and BM25 selected documents
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix[top_k_bm25_indices]).flatten()
    return cosine_similarities, tfidf_matrix

# Function to combine BM25 and TF-IDF results
def combined_search(query, top_k=5, corpus=df["text"]):
    # Perform BM25 search
    top_k_bm25_indices, bm25_scores = bm25_search1(query, top_k, corpus)
    
    # Perform TF-IDF search to re-rank the BM25 results
    cosine_similarities, _ = tfidf_search1(query, top_k_bm25_indices, corpus)
    
    # Combine scores (60% BM25 + 40% TF-IDF)
    bm25_tfidf_scores = [(bm25_scores[idx], cosine_similarities[i]) for i, idx in enumerate(top_k_bm25_indices)]
    
    # Weighted scoring (60% BM25 + 40% TF-IDF)
    final_scores = [(bm25_score * 0.6 + tfidf_score * 0.4) for bm25_score, tfidf_score in bm25_tfidf_scores]
    
    # Sort by final score
    sorted_indices = np.argsort(final_scores)[::-1]

    # Sort top_k_bm25_indices like sorted_indices
    sorted_top_k_bm25_indices = [top_k_bm25_indices[i] for i in sorted_indices]
    
    return sorted_top_k_bm25_indices

# Function to display final results
def display_results(sorted_top_k_bm25_indices, data):
    print("\nTop Matching Podcasts:")
    for idx in sorted_top_k_bm25_indices:
        print(f"{data.iloc[sorted_top_k_bm25_indices[idx]]['title']} by {data.iloc[sorted_top_k_bm25_indices[idx]]['source']}")


# Example Usage
if __name__ == "__main__":
    # Example query and corpus (data should be your pandas DataFrame containing the transcript, title, and source)
    query = cleaned_query
    # Assuming your DataFrame is named `data` and has the 'title', 'transcript', 'source' columns


    # Combine both BM25 and TF-IDF results
    sorted_top_k_bm25_indices,sorted_indices = combined_search(query, top_k=5, corpus=df["text"])

    # Display the results
    display_results(sorted_indices,sorted_top_k_bm25_indices, df)

# %%
#Evaluate the search function

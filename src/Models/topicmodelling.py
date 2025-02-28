#%%
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.models import CoherenceModel

from bertopic import BERTopic
import contractions
from nltk.stem import WordNetLemmatizer
# %%
data = pd.read_csv('/Users/apoorvareddy/Downloads/Academic/DATS6501/data/data.csv')
# %%
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

    # # Join the lemmas into a single string
    # lemmatized_text = ' '.join(lemmas)

    return lemmas

#%%
#clean and lemmatize text
data['text'] = data['text'].apply(clean_text)
data['text'] = data['text'].apply(lemmatize_text)
#%%
# Create Dictionary & Corpus
dictionary = corpora.Dictionary(data["text"])
corpus = [dictionary.doc2bow(text) for text in data["text"]]

#%%
# Train LDA Model
lda_model = gensim.models.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=10)

# Print Topics
topics = lda_model.print_topics(num_words=5)
for topic in topics:
    print(topic)

# %%
def search_lda(query, top_k=5):
    query_tokens = clean_text(query)  # Tokenize query correctly
    tokens = nltk.word_tokenize(query_tokens)
    query_tokens = [t for t in tokens]
    query_bow = dictionary.doc2bow(query_tokens)  # Convert to bag-of-words format

    # Get topic distribution for the query
    query_topics = lda_model.get_document_topics(query_bow)

    # Compute similarity between query topics and document topics
    doc_scores = []
    for idx, doc_bow in enumerate(corpus):
        doc_topics = lda_model.get_document_topics(doc_bow)
        similarity = sum(min(q[1], d[1]) for q in query_topics for d in doc_topics if q[0] == d[0])
        doc_scores.append((idx, similarity))

    # Sort by similarity and return top-k
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    top_results = [data.iloc[idx]["title"] for idx, _ in doc_scores[:top_k]]

    print("\nLDA Search Results:")
    for result in top_results:
        print(f"{result}")

# Example Query
query_text = "Piltdown man hoax, hoax vaccines autism hoax please help me"
search_lda(query_text) 

# %% BERTopic
from sentence_transformers import SentenceTransformer
data = pd.read_csv('/Users/apoorvareddy/Downloads/Academic/DATS6501/data/data.csv')
# Initialize sentence transformer model for embeddings
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# Fit BERTopic model
bertopic_model = BERTopic(nr_topics=50)
topics, probs = bertopic_model.fit_transform(data["text"])

# Display top topics
bertopic_model.get_topic_info()

# %%
def search_bertopic(query, top_k=3):
    query_embedding = sentence_model.encode([query], convert_to_tensor=True)
    
    # Find similar topics
    topic_idx, _ = bertopic_model.find_topics(query, top_n=top_k)

    # Retrieve documents related to topics
    doc_scores = []
    for idx, topic in enumerate(topics):
        if topic in topic_idx:
            doc_scores.append((idx, topic))

    # Sort and retrieve results
    top_results = [data.iloc[idx]["title"] for idx, _ in doc_scores[:top_k]]

    print("\n🔍 BERTopic Search Results:")
    for result in top_results:
        print(f"📌 {result}")

# Example Query
query_text = "AI in medical science"
search_bertopic(query_text)

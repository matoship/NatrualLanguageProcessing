from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import pickle
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import sys
import math

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()


def preprocess_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = text.lower()
    text = word_tokenize(text)
    text = [ps.stem(word) for word in text if not word in stop_words]
    return text


def load_preprocessed_data(file_path):
    df = pd.read_csv(file_path, sep='\t', on_bad_lines='skip')
    df['processed_question2'] = df['question2'].apply(
        lambda x: preprocess_text(x))
    return df


def load_glove_vectors(glove_file):
    with open(glove_file, 'r') as f:
        model = {}
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            model[word] = embedding
        return model


def save_glove_model(glove_model, output_pickle_file):
    with open(output_pickle_file, 'wb') as f:
        pickle.dump(glove_model, f)


def load_glove_model(glove_pickle_file):
    with open(glove_pickle_file, 'rb') as f:
        return pickle.load(f)


def calculate_word_frequencies(documents):
    word_counts = Counter(word for doc in documents for word in doc)
    total_words = sum(word_counts.values())
    return {word: count / total_words for word, count in word_counts.items()}


def compute_sentence_embedding(sentence, model):
    vectors = [model[word] for word in sentence if word in model]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(next(iter(model.values())).shape)


def compute_sentence_embedding_sif(sentence, model, word_frequencies, a=0.001):
    vectors = [model[word] for word in sentence if word in model]
    weights = [a / (a + word_frequencies.get(word, a))
               for word in sentence if word in model]
    if vectors:
        return np.average(vectors, axis=0, weights=weights)
    else:
        return np.zeros(next(iter(model.values())).shape)


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(dict)
        self.tf_idf = defaultdict(dict)
        self.documents = dict()

    def add_document(self, doc_id, document):
        self.documents[doc_id] = document
        words_count = len(document)
        word_freq = Counter(document)

        for word, freq in word_freq.items():
            tf = freq / float(words_count)
            self.index[word][doc_id] = tf

    def compute_tf_idf(self, num_docs):
        for word, docs in self.index.items():
            idf = math.log(num_docs / float(len(docs)))
            for doc_id, tf in docs.items():
                self.tf_idf[word][doc_id] = tf * idf

    def query(self, query, top_k=5):
        scores = defaultdict(float)

        for word in query:
            if word in self.tf_idf:
                for doc_id, tf_idf in self.tf_idf[word].items():
                    scores[doc_id] += tf_idf

        scores_sorted = sorted(
            scores.items(), key=lambda x: x[1], reverse=True)

        return scores_sorted[:top_k]


class SentenceEmbedding:
    def __init__(self, model):
        self.model = model
        self.doc_embeddings = {}

    def add_document(self, doc_id, document):
        self.doc_embeddings[doc_id] = compute_sentence_embedding(
            document, self.model)

    def query(self, query, top_k=5):
        query_vector = compute_sentence_embedding(query, self.model)
        scores = []
        for doc_id, doc_vector in self.doc_embeddings.items():
            cos_sim = cosine_similarity(query_vector.reshape(
                1, -1), doc_vector.reshape(1, -1))[0][0]
            scores.append((doc_id, cos_sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class SentenceEmbeddingAlternative:
    def __init__(self, model, word_frequencies):
        self.model = model
        self.word_frequencies = word_frequencies
        self.doc_embeddings = {}

    def add_document(self, doc_id, document):
        self.doc_embeddings[doc_id] = compute_sentence_embedding_sif(
            document, self.model, self.word_frequencies)

    def query(self, query, top_k=5):
        query_vector = compute_sentence_embedding_sif(
            query, self.model, self.word_frequencies)
        scores = []
        for doc_id, doc_vector in self.doc_embeddings.items():
            cos_sim = cosine_similarity(query_vector.reshape(
                1, -1), doc_vector.reshape(1, -1))[0][0]
            scores.append((doc_id, cos_sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def main():
    # Load preprocessed data
    df = load_preprocessed_data('data.tsv')

    glove_text = load_glove_vectors('glove.6B.100d.txt')
    save_glove_model(glove_text, 'glove_model.pickle')

    # Load GloVe model
    glove_model = load_glove_model('glove_model.pickle')

    # Calculate word frequencies
    word_frequencies = calculate_word_frequencies(
        df['processed_question2'].tolist())

    # Build the inverted index using TF-IDF
    inverted_index_tfidf = InvertedIndex()
    for i, row in df.iterrows():
        inverted_index_tfidf.add_document(
            row['id'], row['processed_question2'])
    inverted_index_tfidf.compute_tf_idf(len(df))

    # Build the sentence embeddings using average embeddings
    sentence_embedding_avg = SentenceEmbedding(glove_model)
    for i, row in df.iterrows():
        sentence_embedding_avg.add_document(
            row['id'], row['processed_question2'])

    # Build the sentence embeddings using SIF embeddings
    sentence_embedding_sif = SentenceEmbeddingAlternative(
        glove_model, word_frequencies)
    for i, row in df.iterrows():
        sentence_embedding_sif.add_document(
            row['id'], row['processed_question2'])

    # Take a query from user
    question = sys.argv[1]
    preprocessed_question = preprocess_text(question)
    # Perform the query and print top 5 results using each method
    print("\nTop 5 matches using TF-IDF:")
    top_tfidf = inverted_index_tfidf.query(preprocessed_question)
    seen_questions = set()
    for doc_id, _ in top_tfidf:
        question = df.loc[df['id'] == doc_id, 'question2'].values[0]
        if question not in seen_questions:
            seen_questions.add(question)
            print(question)

    print("\nTop 5 matches using average embeddings:")
    top_avg = sentence_embedding_avg.query(preprocessed_question)
    seen_questions = set()
    for doc_id, _ in top_avg:
        question = df.loc[df['id'] == doc_id, 'question2'].values[0]
        if question not in seen_questions:
            seen_questions.add(question)
            print(question)

    print("\nTop 5 matches using SIF embeddings:")
    top_sif = sentence_embedding_sif.query(preprocessed_question)
    seen_questions = set()
    for doc_id, _ in top_sif:
        question = df.loc[df['id'] == doc_id, 'question2'].values[0]
        if question not in seen_questions:
            seen_questions.add(question)
            print(question)


if __name__ == "__main__":
    main()

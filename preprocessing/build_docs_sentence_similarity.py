import yaml
import math
import string
import warnings
from tqdm import tqdm
import _pickle as pk

import torch
from torch.utils.data import Dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("punkt")
stopwords = list(stopwords.words("english"))

warnings.filterwarnings("ignore")


class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        tokens = []
        for token in word_tokenize(doc):
            if token not in string.punctuation and token not in stopwords:
                tokens.append(token)

        tokens = [self.wnl.lemmatize(token) for token in tokens]
        return tokens


def get_docs_sents_similarity(full_data, small_data):
    print("[Building Sentences TF-IDF]")
    vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), ngram_range=(1, 2))
    corpus = [sent for doc in full_data for sent in doc]
    vectorizer.fit(corpus)

    docs_sents_similarity = []
    for doc in tqdm(small_data):
        doc_tfidf = vectorizer.transform(doc)
        similarity = cosine_similarity(doc_tfidf, doc_tfidf)

        sent_pair_score = []

        if len(doc) == 1:
            sent_pair = (0, 0)
            score = similarity[0][0]
            sent_pair_score.append((sent_pair, score))

        for i in range(similarity.shape[0]):
            for j in range(i + 1, similarity.shape[0]):
                sent_pair = (i, j)
                score = similarity[i][j]
                sent_pair_score.append((sent_pair, score))

        sent_pair_score.sort(key=lambda x: x[1], reverse=True)
        docs_sents_similarity.append(sent_pair_score)

    return docs_sents_similarity


if __name__ == "__main__":
    with open("config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    with open(config["dataset"]["full_docs_sentence"], "rb") as f:
        full_data = pk.load(f)

    with open(config["dataset"]["docs_sentence"], "rb") as f:
        small_data = pk.load(f)

    docs_sents_similarity = get_docs_sents_similarity(full_data, small_data)
    print("[Saving full documents sentences similarity]")
    with open(config["dataset"]["full_docs_sentence_similarity"], "wb") as f:
        pk.dump(docs_sents_similarity, f)
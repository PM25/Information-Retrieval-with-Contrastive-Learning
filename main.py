#%%
import yaml
from utils import *

with open("config.yaml", "r") as stream:
    config = yaml.safe_load(stream)

all_dataset = TorchDataset(data_dir=config["data_dir"])


# %%
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path

from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    BaggingClassifier,
)

seed = 1009

MIN_DF = 100  # 0.155
MAX_DF = 0.5  # 0.72

models = {
    # "LogisticRegression": lambda: LogisticRegression(random_state=seed),
    # "RandomForestClassifier": lambda: RandomForestClassifier(random_state=seed),
    # "HistGradientBoostingClassifier": lambda: HistGradientBoostingClassifier(
    #     random_state=seed
    # ),
    # "RBF_SVM": lambda: SVC(probability=True, random_state=seed),
    # "LinearSVM": lambda: SVC(kernel="linear", probability=True, random_state=seed),
    # "MLPClassifier": lambda: MLPClassifier(
    #     max_iter=500, warm_start=True, random_state=seed
    # ),
    "LogisticRegressionCV": lambda: LogisticRegressionCV(random_state=seed),
    # "KNeighborsClassifier": lambda: KNeighborsClassifier(n_neighbors=2),
}


def process_data(dataset):
    _input = []
    answer = []

    for datum in dataset:
        article = " ".join(datum["evidences"])
        article = article.lower()
        question = datum["claim"]
        _input.append(question + article)
        answer.append(datum["label"])

    return _input, answer


def train_and_evaluate(corpus, answer, min_df=0.1, max_df=0.8, seed=1009, val_size=0.2):
    if val_size > 0:
        train_x, test_x, train_y, test_y = train_test_split(
            corpus.copy(), answer.copy(), test_size=val_size, random_state=seed
        )
    else:
        train_x = corpus.copy()
        train_y = answer.copy()

    clfs, tfidf_vec = train(train_x, train_y, min_df, max_df)

    preds = predict_class(clfs, tfidf_vec, train_x)
    # train_score = f1_score(train_y, preds)
    train_score = 0

    if val_size > 0:
        preds = predict_class(clfs, tfidf_vec, test_x)
        # val_f1 = f1_score(test_y, preds)
        val_f1 = 0
        print(classification_report(test_y, preds))

        return clfs, (train_score, val_f1)
    else:
        return clfs, (train_score)


def train(corpus, answer, min_df=0.1, max_df=0.5):
    tfidf_vec = TfidfVectorizer(min_df=min_df, max_df=max_df, max_features=5000)

    tfidf_vec.fit(corpus)

    train_x = tfidf_vec.transform(corpus)
    train_y = answer.copy()

    clfs = []
    for name, model in models.items():
        clf = model().fit(train_x, train_y)
        clfs.append(clf)

    weight = list(tfidf_vec.vocabulary_.items())
    for i in range(len(weight)):
        weight[i] = (weight[i][0], round(clfs[0].coef_[0][weight[i][1]], 2))
    print(sorted(weight, key=lambda i: i[1], reverse=True))

    return clfs, tfidf_vec


def predict_prob(classifiers, tfidf_vec, corpus):
    data_x = tfidf_vec.transform(corpus)

    probs = []
    for clf in classifiers:
        prob = clf.predict_proba(data_x)[:, 1]
        probs.append(prob)

    probs = np.array(probs)
    probs = np.mean(probs, axis=0)

    return probs


def predict_class(classifiers, tfidf_vec, corpus):
    preds = predict_prob(classifiers, tfidf_vec, corpus)

    preds[preds > 0.5] = 1
    preds[preds <= 0.5] = 0

    return preds


if __name__ == "__main__":
    corpus, answer = process_data(all_dataset)
    print(corpus[:5])

    print("[Testing with 50 different seeds]")
    train_scores, val_f1s = [], []
    for _ in range(10):
        seed = random.randint(0, 9999)
        clfs, (train_score, val_f1) = train_and_evaluate(
            corpus, answer, MIN_DF, MAX_DF, seed=seed, val_size=0.25
        )
        train_scores.append(train_score)
        val_f1s.append(val_f1)
        print(f"[seed={seed:<4}] train roc: {train_score:.3f} | test f1: {val_f1:.3f}")

    print("=" * 25)
    print(f"average train roc auc score: {np.mean(train_scores):.5f}")
    print(f"average val f1 score: {np.mean(val_f1s):.5f}")
# %%

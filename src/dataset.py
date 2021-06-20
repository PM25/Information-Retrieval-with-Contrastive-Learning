import json
import yaml
import math
import string
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from unicodedata import normalize

import torch
from torch.utils.data import Dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download("wordnet")
stopwords = list(stopwords.words("english"))

warnings.filterwarnings("ignore")


def process_wiki(fname):
    with open(fname, "r") as f:
        wiki = json.load(f)

    for datum in tqdm(wiki.values()):
        lines = []
        for line in datum["lines"].split("\n"):
            line = line.split("\t")
            line = " ".join(line[1:])
            lines.append(line)

        datum["lines"] = lines

    return wiki


def process_trainjsonl(fname):
    with open(fname, "r", encoding="utf-8") as f:
        json_strs = f.readlines()

    out = []

    for json_str in json_strs:
        dic = json.loads(json_str)
        _id = dic["id"]
        claim = dic["claim"]
        label = dic["label"]

        # One sample of clean_evidence
        # {doc_id: [sent_id, sent_id, ...]}
        clean_evidences = {}
        for evidences in dic["evidence"]:
            for evidence in evidences:
                if evidence[2] is not None:
                    doc_id = normalize("NFKD", evidence[2])
                    sent_id = evidence[3]
                    clean_evidences[doc_id] = clean_evidences.get(doc_id, []) + [
                        sent_id
                    ]

        out.append(
            {
                "id": _id,
                "claim": claim,
                "label": label,
                "evidences": clean_evidences,
            }
        )

    return out


class DocDataset(Dataset):
    def __init__(self, data, args):
        super().__init__()
        self.data = data
        self.sample_method = args.sample
        if self.sample_method == "tf_idf":
            with open(
                args.config["dataset"]["full_docs_sentence_similarity"], "rb"
            ) as f:
                self.docs_sents_similarity = pk.load(f)
            self.ratio = 0.1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        doc = self.data[idx]

        if self.sample_method == "uniform":
            sent1, sent2 = np.random.choice(doc, size=2, replace=False)

        elif self.sample_method == "tf_idf":
            doc_sent_similarity = self.docs_sents_similarity[idx]
            k = math.ceil(len(doc_sent_similarity) * self.ratio)
            (i, j), score = random.choice(doc_sent_similarity[:k])
            sent1, sent2 = doc[i], doc[j]

        return torch.LongTensor([idx]), sent1, sent2


class FeverDataset(Dataset):
    def __init__(self, wiki_path, data_path):
        super().__init__()
        self.wiki = process_wiki(wiki_path)

        train_path = Path(data_path) / "train.jsonl"
        train_data = process_trainjsonl(train_path)
        train_data = self.process(train_data)

        self.label_map = {"SUPPORTS": 1, "REFUTES": 0}

        self.data = []
        for datum in train_data:
            label = datum["label"]  # SUPPORTS, NOT VERIFIABLE, REFUTES
            if label == "NOT ENOUGH INFO":
                continue

            self.data.append(
                {
                    "id": datum["id"],
                    "label": self.label_map[label],
                    "claim": datum["claim"],
                    "evidences": datum["evidences"],
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def process(self, data):
        data = data.copy()
        for datum in data:
            evidences = datum["evidences"]
            process_evidences = []

            for doc_id, sent_ids in evidences.items():
                process_evidences.append(
                    {
                        "title": doc_id,
                        "document": self.wiki[doc_id]["lines"],
                        "sent_idx": sent_ids,
                    }
                )
            datum["evidences"] = process_evidences

        return data


def get_dataloader(data, args, train=True):
    bsz = (
        args.config["train"]["batch_size"]
        if train
        else args.config["eval"]["batch_size"]
    )
    n_jobs = args.config["train"]["n_jobs"] if train else args.config["eval"]["n_jobs"]

    dataset = DocDataset(data, args)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=bsz,
        shuffle=train,
        num_workers=n_jobs,
        drop_last=train,
        pin_memory=True
        # collate_fn=dataset.collate_fn
    )
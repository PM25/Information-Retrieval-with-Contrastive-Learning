import json
import yaml
import math
import string
import random
import warnings
import numpy as np
import pandas as pd
import _pickle as pk
from tqdm import tqdm
from pathlib import Path
from unicodedata import normalize

import torch
from torch.utils.data import Dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing.docs_sentence_extraction import sentence_cleaning


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


def process_jsonl(fname):
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
    def __init__(self, args):
        super().__init__()
        with open(args.config["dataset"]["docs_sentence"], "rb") as f:
            self.data = pk.load(f)
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
    def __init__(self, wiki_path, fever_path):
        super().__init__()
        self.wiki = process_wiki(wiki_path)

        fever_data = process_jsonl(fever_path)
        fever_data = self.process(fever_data)

        self.label_map = {"SUPPORTS": 1, "REFUTES": 0}

        self.data = []
        for datum in fever_data:
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

    def parsing(self):
        for d in self.data:
            evidences = {}
            for doc in d["evidences"]:
                evidences[doc["title"]] = []
                sent_idx = np.unique(doc["sent_idx"])
                for idx in sent_idx:
                    lines = sentence_cleaning(doc["document"][idx])
                    evidences[doc["title"]].append((idx, lines))
            d["evidences"] = evidences

    def collate_fn(self, data):
        return data

def get_dataloader(args, train=True):
    bsz = (
        args.config["train"]["batch_size"]
        if train
        else args.config["eval"]["batch_size"]
    )
    n_jobs = args.config["train"]["n_jobs"] if train else args.config["eval"]["n_jobs"]

    if args.data == 'doc':
        dataset = DocDataset(args)
        collate_fn = None
    elif args.data == 'fever':
        dataset = FeverDataset(args.config["dataset"]["small_wiki"], args.config["dataset"]["dev_data"])
        collate_fn = dataset.collate_fn

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=bsz,
        shuffle=train,
        num_workers=n_jobs,
        drop_last=train,
        pin_memory=True,
        collate_fn=collate_fn
    )

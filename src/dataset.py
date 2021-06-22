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
from transformers import AutoTokenizer

import torch
from torch.utils.data import Dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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


class FeverDatasetTokenize(Dataset):
    def __init__(self, wiki_path, data_path):
        super().__init__()
        self.wiki = process_wiki(wiki_path)
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.label_map = {"SUPPORTS": 1, "REFUTES": 0}

        train_path = Path(data_path) / "train.jsonl"
        train_data = process_trainjsonl(train_path)
        train_data = self.process(train_data)

        pos_data, neg_data = [], []
        for datum in train_data:
            if datum["label"] == "NOT ENOUGH INFO":
                continue

            label = self.label_map[datum["label"]]

            if label == 1:
                pos_data.append(datum)
            elif label == 0:
                neg_data.append(datum)

        print(len(pos_data), len(neg_data))

        # balance pos & neg data
        if len(pos_data) > len(neg_data):
            pos_data = random.sample(pos_data, k=len(neg_data))
        elif len(pos_data) < len(neg_data):
            neg_data = random.sample(neg_data, k=len(pos_data))

        assert len(pos_data) == len(neg_data)
        train_data = pos_data + neg_data
        random.shuffle(train_data)
        print(f"total training data: {len(train_data)}")

        self.data = []
        for datum in train_data:
            label = datum["label"]  # SUPPORTS, NOT VERIFIABLE, REFUTES

            self.data.append(
                {
                    "id": datum["id"],
                    "label": self.label_map[label],
                    "claim": datum["claim"],
                    "evidences": datum["evidences"],
                    "input_ids": datum["input_ids"],
                    "attention_mask": datum["attention_mask"],
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def process(self, data):
        data = data.copy()

        for datum in tqdm(data):
            evidences = datum["evidences"]

            process_evidences = []
            for doc_id, sent_ids in evidences.items():
                process_evidences.extend(doc_id.split("_"))
                for sent_id in sent_ids:
                    doc = self.wiki[doc_id]["lines"]
                    process_evidences.append(doc[sent_id])

            claim = datum["claim"]
            evidences = " ".join(process_evidences)

            tokenize_data = self.tokenizer(
                claim,
                evidences,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                max_length=512,
                return_tensors="pt",
            )
            datum["evidences"] = evidences
            datum["input_ids"] = tokenize_data["input_ids"].flatten()
            datum["attention_mask"] = tokenize_data["attention_mask"].flatten()

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
import os
import json
import yaml
import string
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from unicodedata import normalize
from transformers import AutoTokenizer

import torch
from torch.utils.data import Dataset

from elastic_search.search import batch_search

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def process_wiki(fname):
    with open(fname, "r") as f:
        wiki = json.load(f)

    for datum in tqdm(wiki.values(), desc="[Processing Wiki Data]"):
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


class FeverDatasetTokenize(Dataset):
    def __init__(self, wiki_path, data_path):
        super().__init__()
        self.wiki = process_wiki(wiki_path)
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.label_map = {"SUPPORTS": 1, "REFUTES": 0}

        train_path = Path(data_path)
        train_data = process_trainjsonl(train_path)
        train_data = self.process(train_data)

        self.data = []
        for datum in train_data:
            if datum["label"] == "NOT ENOUGH INFO":
                continue

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
        for datum in tqdm(data, desc="[Tokenizing Data]"):
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


class EvaluationFeverDataset(Dataset):
    def __init__(self, wiki_path, data_path, method="tf_idf"):
        super().__init__()
        self.wiki = process_wiki(wiki_path)
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.label_map = {"SUPPORTS": 1, "REFUTES": 0}
        self.method = method

        train_path = Path(data_path)
        train_data = process_trainjsonl(train_path)

        if self.method == "tf_idf":
            claims = [datum["claim"] for datum in train_data]
            num_batches = (len(claims) - 1) // 1000 + 1

            retrieves = []
            for i in tqdm(range(num_batches), desc="[Retrieve Evidences with TF-IDF]"):
                batch = claims[i * 1000 : (i + 1) * 1000]
                retrieves += batch_search(batch, count=15, max_threads=32)
            assert len(retrieves) == len(claims)

            for datum, evidences in zip(train_data, retrieves):
                datum["pred_evidences"] = evidences

        train_data = self.process(train_data)

        self.data = []
        for datum in train_data:
            if datum["label"] == "NOT ENOUGH INFO":
                continue

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
        for datum in tqdm(data, desc="[Tokenizing Data]"):
            claim = datum["claim"]
            datum["original_evidences"] = datum["evidences"]

            if self.method == "tf_idf":
                process_evidences = []
                for retrieve in datum["pred_evidences"]:
                    process_evidences.append(retrieve["title"])
                    process_evidences.append(retrieve["text"])
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
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
                    if doc_id not in clean_evidences:
                        clean_evidences[doc_id] = {sent_id}
                    else:
                        clean_evidences[doc_id].add(sent_id)

        for doc_id in clean_evidences:
            # transform set to list
            evidences = list(clean_evidences[doc_id])
            clean_evidences[doc_id] = sorted(evidences)

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
    def __init__(
        self, wiki_path, data_path, process_path=None, method="tf_idf", max_evidences=10
    ):
        super().__init__()
        self.wiki = process_wiki(wiki_path)
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.label_map = {"SUPPORTS": 1, "REFUTES": 0}
        self.method = method
        self.max_evidences = max_evidences
        self.process_path = process_path

        train_path = Path(data_path)
        train_data = process_trainjsonl(train_path)
        train_data = [
            datum for datum in train_data if datum["label"] != "NOT ENOUGH INFO"
        ]

        if self.method == "tf_idf":
            claims = [datum["claim"] for datum in train_data]
            retrieves = batch_search(claims, count=self.max_evidences, max_threads=32)

            for datum, evidences in zip(train_data, retrieves):
                out = []
                for _id, retrieve in evidences:
                    title = retrieve["title"].replace(" ", "_")
                    _id = eval(_id.split("_")[1])
                    out.append((title, _id))
                datum["pred_evidences"] = out

        elif self.process_path != None:
            with open(self.process_path, "r") as f:
                process_data = json.load(f)

            process_data = {datum["id"]: datum for datum in process_data}

            for datum in train_data:
                _id = datum["id"]
                datum["evidences"] = process_data[_id]["evidences"]
                datum["pred_evidences"] = process_data[_id]["k_evidences"]

        self.train_data = self.process(train_data)

        self.data = []
        for datum in self.train_data:
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
            _id = datum["id"]
            claim = datum["claim"]
            datum["truth_evidences"] = datum["evidences"]

            if self.process_path != None:
                evidences = self.process_evidence(datum["evidences"])
            elif self.method == "tf_idf":
                evidences = self.process_tf_idf_evidence(datum["pred_evidences"])
            elif self.method == "uniform":
                evidences = self.uniform_sample(k=self.max_evidences)
            elif self.method == "truth":
                evidences = self.process_evidence(datum["evidences"])
            elif self.method == "claim_only":
                evidences = ""

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

    def process_evidence(self, evidences):
        process_evidences = []
        for doc_id, sent_ids in evidences.items():
            process_evidences.extend(doc_id.split("_"))
            for sent_id in sent_ids:
                doc = self.wiki[doc_id]["lines"]
                process_evidences.append(doc[sent_id])

        return " ".join(process_evidences)

    def process_tf_idf_evidence(self, evidences):
        process_evidences = {}
        for title, _id in evidences:
            doc = self.wiki[title]["lines"]
            sent = (_id, doc[_id])
            process_evidences[title] = process_evidences.get(title, []) + [sent]

        combine_evidences = []
        for doc_id, sent_ids in process_evidences.items():
            combine_evidences.append(doc_id)
            for _id, sent in sorted(sent_ids, key=lambda x: x[0]):
                combine_evidences.append(sent)

        return " ".join(combine_evidences)

    def uniform_sample(self, k=10):
        sents = []
        count = 0
        while count < k:
            doc_id = random.choice(list(self.wiki))
            lines = self.wiki[doc_id]["lines"]

            doc_id = doc_id.replace("_", " ")
            sents.append(doc_id)
            lines = random.sample(lines, k=random.randint(1, len(lines)))
            sents.extend(lines)
            count += len(lines)

        return " ".join(sents[:count])
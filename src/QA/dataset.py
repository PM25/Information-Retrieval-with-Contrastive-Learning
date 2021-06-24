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

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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


class FeverDatasetTokenize(Dataset):
    def __init__(self, wiki_path, data_path):
        super().__init__()
        self.wiki = process_wiki(wiki_path)
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.label_map = {"SUPPORTS": 1, "REFUTES": 0}

        train_path = Path(data_path) / "train.jsonl"
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
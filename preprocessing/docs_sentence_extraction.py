import yaml
import json
import argparse
import re, string
from pathlib import Path

import _pickle as pk


def sub(m):
    return (
        ""
        if m.group() in {"-LRB-", "-RRB-", "-LSB-", "-RSB-", "''", "``", "--"}
        else m.group()
    )


def extract_docs_sentence(dic):
    docs = []

    for doc_string in dic.values():
        # get the number of sentences in the document
        length = doc_string["lines"][-3:].strip()
        if not length.isdigit():
            continue

        length = int(length)
        if length <= 2:
            continue

        # remove the above redundancy words
        lines = re.sub(r"[^ ]+", sub, doc_string["lines"])

        doc = []
        for i in range(length):

            # get the start position of the sentence
            s = lines[lines.find("%d\t" % (i)) :]

            # get the end position of the sentence (rule: the end of the sentence should be 1). ".\t", 2). ".\n", 3). "(i+1)\t"
            end_pos = min(
                [
                    e
                    for e in [s.find(x) for x in [".\t", ".\n", "%d\t" % (i + 1)]]
                    if e > 0
                ]
            )

            s = s[len(str(i)) : end_pos].strip() + "."

            # if len(s) == 1: empty string
            if len(s) == 1:
                continue
            doc.append(s)
        docs.append(doc)

    return docs


if __name__ == "__main__":
    with open("config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    # Small wiki documents sentences extraction
    print("[Saving small wiki sentences extraction]")
    docs_sentence_path = Path(config["dataset"]["docs_sentence"])
    if not docs_sentence_path.is_file():
        with open(config["dataset"]["small_wiki"], "r", encoding="utf-8") as f:
            d = json.load(f)
        docs = extract_docs_sentence(d)

        docs_sentence_path.parent.mkdir(parents=True, exist_ok=True)
        with open(docs_sentence_path, "wb") as f:
            pk.dump(docs, f)

    # Full wiki documents sentences extraction
    print("[Saving full wiki sentences extraction]")
    full_docs_sentence_path = Path(config["dataset"]["full_docs_sentence"])
    if not full_docs_sentence_path.is_file():
        with open(config["dataset"]["full_wiki"], "r", encoding="utf-8") as f:
            d = json.load(f)
        docs = extract_docs_sentence(d)

        full_docs_sentence_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_docs_sentence_path, "wb") as f:
            pk.dump(docs, f)

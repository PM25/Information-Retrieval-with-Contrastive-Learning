import yaml
import json
import re, string

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
        length = int(doc_string["lines"][-3:].strip())
        lines = re.sub(r"[^ ]+", sub, doc_string["lines"])

        doc = []
        for i in range(length):
            s = lines[lines.find("%d\t" % (i)) :]
            end_pos = min(
                [
                    e
                    for e in [s.find(x) for x in [".\t", ".\n", "%d\t" % (i + 1)]]
                    if e > 0
                ]
            )

            s = s[len(str(i)) : end_pos].strip() + "."
            if len(s) == 1:
                continue
            doc.append(s)
        docs.append(doc)
    return docs


if __name__ == "__main__":
    with open("config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    with open(config["dataset"]["small_wiki"], "r", encoding="utf-8") as f:
        d = json.load(f)
    docs = extract_docs_sentence(d)
    with open(config["dataset"]["docs_sentence"], "wb") as f:
        pk.dump(docs, f)

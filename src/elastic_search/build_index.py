import bz2
import json
import yaml
import socket
import numpy as np
from tqdm import tqdm
from pathlib import Path

from elasticsearch import Elasticsearch
from elasticsearch.helpers import parallel_bulk


def extract_data(data, index):
    for doc_title, datum in data.items():
        doc_title = doc_title.replace("_", " ")
        for idx, line in enumerate(datum["lines"].split("\n")):
            line = " ".join(line.split("\t")[1:])
            if len(line) == 0:
                continue

            yield {
                "_index": index,
                "_type": "doc",
                "_id": f"{doc_title}_{idx}",
                "_source": {
                    "title": doc_title,
                    "text": line,
                },
            }


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


if __name__ == "__main__":
    with open("config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
        dataset = config["dataset"]
        config = config["elastic_search"]

    print(f"[Using Port: {config['port']}]")
    # assert is_port_in_use(config["port"]) == False
    es = Elasticsearch(hosts=["localhost"], port=config["port"])

    # ignore 400 cause by IndexAlreadyExistsException when creating an index
    es.indices.delete(index=config["index"], ignore=[400, 404])
    status = es.indices.create(index=config["index"], ignore=400)
    print(status)

    print("[Loading Wiki]")
    with open(dataset["full_wiki"], "r") as f:
        wiki = json.load(f)
    print("[Wiki Loaded]")

    for success, info in tqdm(
        parallel_bulk(
            es,
            extract_data(wiki, config["index"]),
            doc_type="doc",
            thread_count=config["num_threads"],
            request_timeout=30,
        ),
        desc="[Indexing]",
        total=len(wiki)
    ):
        assert success
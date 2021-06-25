import json
import yaml
from pathlib import Path

from elasticsearch import Elasticsearch

with open("config.yaml", "r") as stream:
    config = yaml.safe_load(stream)
    config = config["elastic_search"]

# Search documents by elasticsearch.
_es = Elasticsearch(hosts=["localhost"], port=config["port"], timeout=240)
_methods = {
    "match": lambda query: {"match": {"text": query}},
    "multi_match": lambda query: {
        "multi_match": {
            "query": query,
            "fields": ["title", "text"],
        }
    },
    "multi_match2": lambda query: {
        "multi_match": {"query": query, "fields": ["title^1.25", "text"]}
    },
}


def search(query, method="multi_match", count=10):
    search_method = _methods[method]

    res = _es.search(
        index=config["index"],
        doc_type="doc",
        body={
            "size": count,
            "query": search_method(query),
            "sort": [{"_score": {"order": "desc"}}, {"_id": {"order": "asc"}}],
            "_source": {"includes": ["title", "text"]},
        },
    )

    return [doc["_source"] for doc in res["hits"]["hits"]]


def search_title(query, method="multi_match", count=10):
    return [r["title"] for r in search(query, method, count)]


if __name__ == "__main__":
    print(search("Hello World"))
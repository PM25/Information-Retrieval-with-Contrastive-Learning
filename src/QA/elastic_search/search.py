import json
import yaml
from tqdm import tqdm
from pathlib import Path

from elasticsearch import Elasticsearch

with open("config.yaml", "r") as stream:
    config = yaml.safe_load(stream)
    config = config["elastic_search"]

# Search documents by elasticsearch.
_es = Elasticsearch(hosts=["localhost"], port=config["port"], timeout=300)
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


def multi_search(
    queries, method="multi_match", count=10, max_threads=10, include_id=True
):
    search_method = _methods[method]

    body_request = []
    for query in queries:
        req_head = {"index": config["index"], "type": "doc"}
        req_body = {
            "size": count,
            "query": search_method(query),
            "sort": [{"_score": {"order": "desc"}}, {"_id": {"order": "asc"}}],
            "_source": {"includes": ["title", "text"]},
        }
        body_request.extend([req_head, req_body])

    res = _es.msearch(
        index=config["index"],
        doc_type="doc",
        body=body_request,
        max_concurrent_searches=max_threads,
    )

    out = []
    for r in res["responses"]:
        if include_id:
            out.append([(doc["_id"], doc["_source"]) for doc in r["hits"]["hits"]])
        else:
            out.append([doc["_source"] for doc in r["hits"]["hits"]])

    assert len(out) == len(queries)

    return out


def batch_search(
    queries, count=10, method="multi_match", batch_size=1000, max_threads=10
):
    num_batches = (len(queries) - 1) // batch_size + 1

    retrieves = []
    for i in tqdm(range(num_batches), desc="[Batch Search Documents]"):
        batch = queries[i * batch_size : (i + 1) * batch_size]
        retrieves += multi_search(
            batch, count=count, method=method, max_threads=max_threads
        )
    assert len(retrieves) == len(queries)

    return retrieves


def search_title(query, method="multi_match", count=10):
    return [r["title"] for r in search(query, method, count)]


if __name__ == "__main__":
    # print(search("Hello World"))
    print(batch_search(["hello", "world", "Jason"], count=2))

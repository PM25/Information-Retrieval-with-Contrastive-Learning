from tqdm import tqdm
import json
from src.model import load_model
from src.dataset import FeverDataset 
import numpy as np
import _pickle as pk
import random
import torch
import os
from preprocessing.drqa.retriever.utils import load_sparse_csr, filter_ngram, hash
from preprocessing.drqa import tokenizers
from preprocessing.docs_sentence_extraction import sentence_extraction, sentence_cleaning
from collections import defaultdict

from torch.utils.data import DataLoader

OOM_RETRY_LIMIT = 10


def evaluate(dataloader, model):
    device = next(model.parameters()).device
    torch.cuda.empty_cache()
    model.eval()

    loss_sum = 0
    oom_counter = 0
    n_sample = 0

    for batch in tqdm(dataloader, desc="Iteration"):
        with torch.no_grad():
            try:
                # load data and compute loss
                indexes, anchor_sample, positive_sample = batch
                anchor_sample, positive_sample = (
                    anchor_sample.to(device),
                    positive_sample.to(device),
                )
                loss_sum += model(anchor_sample, positive_sample).item()

                batch_size = len(indexes)

                # compute n_sample
                n_sample += batch_size

            except RuntimeError as e:
                if not "CUDA out of memory" in str(e):
                    raise
                if oom_counter >= OOM_RETRY_LIMIT:
                    oom_counter = 0
                    break
                oom_counter += 1
                torch.cuda.empty_cache()

    model.train()
    torch.cuda.empty_cache()

    loss_avg = loss_sum / n_sample
    return loss_avg

# load the following data before using the function 
#     matrix, metadata = load_sparse_csr(args.config["dataset"]["tfidf"])
#     count_matrix, _ = load_sparse_csr(args.config["dataset"]["inverted_file"])
#     with open(args.config["dataset"]["full_docs_dict"], "rb") as f:
#         full_docs_dict = pk.load(f)
    
def documents_filtering(claim, args, count_matrix, metadata, bigram_only=True):
    tokenizer = tokenizers.get_class(metadata['tokenizer'])()
    ngrams = metadata['ngram']
    hash_size = metadata['hash_size']

    tokens = tokenizer.tokenize(claim)
    res = tokens.ngrams(n=ngrams, uncased=True,filter_fn=filter_ngram)
    
    if bigram_only:
        bi_res = [t for t in res if len(t.split()) > 1]
        res = res if bi_res == 0 else bi_res

    wids = [hash(w, hash_size) for w in res]

    wids_unique, wids_cnt = np.unique(wids, return_counts=True)

    term, idx = count_matrix[wids_unique].nonzero()
    idx = np.unique(idx)
    
    docs = [metadata["doc_dict"][1][i] for i in idx]
    return docs

def load_total_docs(args):
    dataset = FeverDataset(args.config["dataset"]["small_wiki"], args.config["dataset"]["dev_data"])
    
    matrix, metadata = load_sparse_csr(args.config["dataset"]["tfidf"])
    count_matrix, _ = load_sparse_csr(args.config["dataset"]["inverted_file"])
    with open(args.config["dataset"]["full_docs_dict"], "rb") as f:
        full_docs_dict = pk.load(f)

    total_docs = []
    for d in tqdm(dataset, desc="Retrieve Documents"):
        total_docs.append(documents_filtering(d["claim"], args, count_matrix, metadata, True))

    with open('total_docs.pkl', 'wb') as f:
        pk.dump(total_docs, f)

    total_docs_dict = {}
    claim_docs_dict = {}
    for d, docs in tqdm(zip(dataset, total_docs), total=len(dataset)):
        claim_docs_dict[d["claim"]] = []
        for docs_id in docs:
            if docs_id in full_docs_dict:
                claim_docs_dict[d["claim"]].append(docs_id)
                if docs_id not in total_docs_dict:
                    total_docs_dict[docs_id] = full_docs_dict[docs_id]
    with open('total_docs_dict.pkl', 'wb') as f:
        pk.dump(total_docs_dict, f)
    with open('claim_docs_dict.pkl', 'wb') as f:
        pk.dump(claim_docs_dict, f)

def embedding_transformation(model, sentences, output_path, device):
    dloader = DataLoader(sentences, batch_size=64, shuffle=False, num_workers=8)

    res = []
    model.eval()
    with torch.no_grad():
        for sents in tqdm(dloader, desc="Embedding Transformation"):
            res.append(model.ctx2vec(sents, device).detach_().cpu().numpy())
            torch.cuda.empty_cache()

    res = np.concatenate(res, axis=0).reshape(len(sentences), -1)
    sent2embedding_dict = dict(zip(sentences, res))

    with open(output_path, 'wb') as f:
        pk.dump(sent2embedding_dict, f)

    return sent2embedding_dict

# The data should at least follow the below format:
# [
#     {   
#         "evidences": {
#             "doc_id": [
#                 ("sent_id", "sent"), ...
#             ],
#             "doc_id": [
#                 ("sent_id", "sent"), ...
#             ]
#         }
#     },
# ]

def all_setences_extraction(data):
    all_sentences = [] 
    reverse_dict_list = []
    for d in tqdm(data, desc="Sentences Extraction"):
        r_dict = {}
        for title, contents in d["evidences"].items():
            contents = [c for i, c in contents]
            r_dict.update(dict(zip(contents, [title]*len(contents))))
            all_sentences += contents
        reverse_dict_list.append(r_dict)
    all_sentences = np.unique(all_sentences)
    return all_sentences, reverse_dict_list

def predict(args, data):
    _, model, _, _ = load_model(args.ckpt)
    model = model.to(args.device)

    all_sentences, reverse_dict_list = all_setences_extraction(data)

    try:
        with open(args.config["dataset"]["sent2embedding_dict"], 'rb') as f:
            sent2embedding_dict = pk.load(f)
    except:
        sent2embedding_dict = embedding_transformation(model, all_sentences, args.config["dataset"]["sent2embedding_dict"], args.device)


    json_list = []
    for i, d in tqdm(enumerate(data), desc="Result Mapping", total=len(data)):
        json_dict = {}
        with torch.no_grad():
            claim_vec = model.ctx2vec(d["claim"], args.device).detach_().cpu().numpy().squeeze()
        sents = [s for did, sid, s in d["k_evidences"]]
        sents_vec = np.array([sent2embedding_dict[sent] for sent in sents]).reshape(len(d["k_evidences"]), -1)

        score = np.einsum('j,kj->k', claim_vec, sents_vec)

        rank = score.argsort()[::-1]
        res = np.array(d["k_evidences"])[rank].tolist()

        title_contents_dict = defaultdict(list)
        for doc_id, sent_id, evidence in res:
            title_contents_dict[reverse_dict_list[i][evidence]].append((sent_id, evidence))
        json_dict["id"] = d["id"]
        json_dict["claim"] = d["claim"]
        json_dict['evidences'] = dict(title_contents_dict)
        json_dict['k_evidences'] = res

        json_list.append(json_dict)
    
    with open(args.config["eval"]["output_file"], "w") as f:
        json.dump(json_list, f, indent=4)

def get_cos_similarity(args):
    _, model, _, _ = load_model(args.ckpt)
    model = model.to(args.device)

    dataset = FeverDataset(args.config["dataset"]["small_wiki"], args.config["dataset"]["dev_data"])
    dataset.parsing()

    all_sentences, reverse_dict_list = all_setences_extraction(dataset)
    print(len(all_sentences))
    try:
        with open(args.config["dataset"]["ground_truth_embedding_dict"], 'rb') as f:
            ground_truth_embedding_dict = pk.load(f)
    except:
        ground_truth_embedding_dict = embedding_transformation(model, all_sentences, args.config["dataset"]["ground_truth_embedding_dict"], args.device)
    
    total_scores = 0.0
    for d in tqdm(dataset, desc="Calculating"):
        with torch.no_grad():
            claim_vec = model.ctx2vec(d["claim"], args.device).detach_().cpu().numpy().squeeze()

        for title, contents in d["evidences"].items():
            contents = [c for i, c in contents]
            sents_vec = np.array([ground_truth_embedding_dict[sent] for sent in contents]).reshape(len(contents), -1)
        score = np.einsum('j,kj->k', claim_vec, sents_vec).mean()
        total_scores += score
    print('average cos similarity score: %.4f' % (total_scores / len(dataset)))



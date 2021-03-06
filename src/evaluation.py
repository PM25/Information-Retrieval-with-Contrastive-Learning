from tqdm import tqdm
from src.model import load_model
from src.dataset import get_dataloader
import numpy as np
import _pickle as pk
import random
import torch
import os
from preprocessing.drqa.retriever.utils import load_sparse_csr, filter_ngram, hash
from preprocessing.drqa import tokenizers
from preprocessing.docs_sentence_extraction import sentence_extraction

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


def documents_filtering(claim, args, count_matrix, metadata, full_doc_dict, bigram_only=True):
    tokenizer = tokenizers.get_class(metadata['tokenizer'])()
    ngrams = metadata['ngram']
    hash_size = metadata['hash_size']

    tokens = tokenizer.tokenize(claim)
    res = tokens.ngrams(n=ngrams, uncased=True,filter_fn=filter_ngram)
    
    if bigram_only:
        res = [t for t in res if len(t.split()) > 1]

    wids = [hash(w, hash_size) for w in res]

    wids_unique, wids_cnt = np.unique(wids, return_counts=True)

    term, idx = count_matrix[wids_unique].nonzero()
    idx = np.unique(idx)
    
    docs = {}
    for i in idx:
        # doc_dict[1]: idx to doc_id
        doc_id = metadata["doc_dict"][1][i]
        try:
            docs[doc_id] = full_doc_dict[doc_id]
        except:
            continue
    return docs


def predict(args):
    assert not args.ckpt is None
    _, model, _, _ = load_model(args.ckpt)
    model = model.to(args.device)
    fever_loader = get_dataloader(args, train=False)
    
    matrix, metadata = load_sparse_csr(args.config["dataset"]["tfidf"])
    count_matrix, _ = load_sparse_csr(args.config["dataset"]["inverted_file"])
    with open(args.config["dataset"]["full_docs_dict"], "rb") as f:
        full_docs_dict = pk.load(f)
    import time
    with torch.no_grad():
        for batch in tqdm(fever_loader, desc="Iteration"):
            claim = [data['claim'] for data in  batch]
            s = time.time()
            docs = documents_filtering(claim[0], args, count_matrix, metadata, full_docs_dict, False)
            e = time.time()
            print(e-s)
            print(len(docs))
            # # claim = [data['evidences'][0]['document'][1] for data in  batch]
            # evdn = [data['evidences'][0]['document'][1] for data in  batch]
            # print(claim[0])
            # print(evdn[0])
            
            # clm_vec = model.ctx2vec(claim, args.device)
            # evdn_vec = model.ctx2vec(evdn, args.device)
            # print((clm_vec * evdn_vec).sum(dim=-1).mean())
            # random.shuffle(evdn)
            # evdn_vec = model.ctx2vec(evdn, args.device)
            # print((clm_vec * evdn_vec).sum(dim=-1).mean())
            # print('----------------------------------')
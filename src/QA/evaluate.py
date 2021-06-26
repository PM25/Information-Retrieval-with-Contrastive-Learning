import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu

import torch
from torch.utils.data import DataLoader
from transformers import logging

from model import RoBertaClassifier
from dataset import EvaluationFeverDataset
from setting import get_device

from sklearn.metrics import f1_score, accuracy_score, classification_report

with open("config.yaml", "r") as stream:
    config = yaml.safe_load(stream)
    data = config["dataset"]
    config = config["QA"]

torch_device = get_device(config["device_id"])
torch.cuda.empty_cache()
logging.set_verbosity(logging.ERROR)


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    model.to(torch_device)

    truth, preds = [], []
    for batch in tqdm(val_loader, desc="[Evaluating]"):
        input_ids = batch["input_ids"].to(torch_device)
        attention_mask = batch["attention_mask"].to(torch_device)
        answer = batch["label"].to(torch_device)

        pred, loss = model(input_ids, attention_mask, answer)
        truth.extend(answer.tolist())
        preds.extend(pred.tolist())

    return classification_report(truth, preds)


def process_tf_idf_retrieve(self, retrieves):
    out = []
    for k, (_id, retrieve) in enumerate(retrieves, 1):
        title = retrieve["title"]
        _id = eval(_id.split("_")[1])
        out.append((title, _id))
    return out


def retrieve_recall(pred_dataset):
    k_recalls = {}

    for idx in tqdm(range(len(pred_dataset)), desc="[Evaluating Recall Score]"):
        data = pred_dataset.train_data[idx]

        truth = data["truth_evidences"]
        # truth = {k.replace("_", " "): v for k, v in truth.items()}
        preds = data["pred_evidences"]

        num_truth = sum([len(sent_ids) for sent_ids in truth.values()])
        found = 0

        for k, (title, _id) in enumerate(preds, 1):
            if title in truth:
                if _id in truth[title]:
                    found += 1

            recall = found / num_truth

            if k not in k_recalls:
                k_recalls[k] = []

            k_recalls[k].append(recall)

    k_recall = []
    for k in sorted(k_recalls.keys()):
        avg_recall = sum(k_recalls[k]) / len(k_recalls[k])
        k_recall.append((k, avg_recall))

    return k_recall


def draw_recall(recalls, label="tf-idf"):
    x, y = [], []
    for step, recall in recalls:
        x.append(step)
        y.append(recall)

    plt.plot(x, y, label=label)
    plt.xlabel("k")
    plt.xlabel("Recall")


# def save_tfidf():
#     t = test_dataset.train_data

#     out = []
#     for i in range(len(test_dataset)):
#         datum = test_dataset[i]
#         _id = datum["id"]
#         claim = datum["claim"]
#         preds = t[i]["pred_evidences"]
#         # print(_id, claim, preds)
#         k_e = []
#         for pred in preds:
#             _id = eval(pred[0].split("_")[1])
#             text = pred[1]["text"]
#             title = pred[1]["title"].replace(" ", "_")
#             k_e.append((title, _id, text))

#         process_evidences = {}
#         for _id, retrieve in preds:
#             title = retrieve["title"].replace(" ", "_")
#             _id = eval(_id.split("_")[1])
#             sent = (_id, retrieve["text"])
#             process_evidences[title] = process_evidences.get(title, []) + [sent]

#         out.append(
#             {
#                 "id": _id,
#                 "claim": claim,
#                 "k_evidences": k_e,
#                 "evidences": process_evidences,
#             }
#         )

#     import json

#     with open("/tmp2/py/data/fever/tfidf_retrieve.json", "w") as f:
#         json.dump(out, f)


if __name__ == "__main__":
    qa_model = torch.load(config["save"])

    # evaluation on dev data
    # contrastive_dataset = EvaluationFeverDataset(
    #     data["small_wiki"], data["dev_data"], process_path=".."
    # )

    # contrastive_loader = DataLoader(
    #     contrastive_dataset,
    #     batch_size=config["eval"]["batch_size"],
    #     num_workers=config["eval"]["n_jobs"],
    # )

    # # QA Evaluate
    # report = evaluate(qa_model, contrastive_loader)
    # print("[QA Dev Report]")
    # print(report)

    # IR Evaluate
    tfidf_dataset = EvaluationFeverDataset(
        data["full_wiki"], data["dev_data"], method="tf_idf", max_evidences=100
    )

    # contrastive_recalls = retrieve_recall(contrastive_dataset)
    tfidf_recalls = retrieve_recall(tfidf_dataset)
    print("[IR Dev Report]")
    # draw_recall(contrastive_recalls, "contrastive")
    draw_recall(tfidf_recalls, "tf-idf")
    plt.legend()
    plt.savefig("recall.jpg")

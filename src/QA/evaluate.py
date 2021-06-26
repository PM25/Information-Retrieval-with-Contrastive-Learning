import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path

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


if __name__ == "__main__":
    qa_model = torch.load(config["save"])

    # evaluation on dev data
    test_dataset = EvaluationFeverDataset(
        data["small_wiki"], data["dev_data"], method="uniform"
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["eval"]["batch_size"],
        num_workers=config["eval"]["n_jobs"],
    )
    report = evaluate(qa_model, test_loader)
    print("[Dev Report]")
    print(report)

import yaml
import argparse

import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import logging

from model import RoBertaClassifier
from dataset import FeverDatasetTokenize
from setting import get_device

from sklearn.metrics import f1_score, accuracy_score, classification_report

def get_args():
    # contruct parser object
    parser = argparse.ArgumentParser(description="Argument Parser.")

    parser.add_argument(
        "--config",
        type=str,
        help="Path to experiment configuration.",
        default="config.yaml",
    )
    parser.add_argument(
        "--data_path", default="data/fever/shared_task_dev.jsonl", type=str, help="Directory for testing data."
    )

    # Options
    parser.add_argument(
        "--seed", default=1337, type=int, help="Random seed for reproducable results."
    )
    parser.add_argument(
        "--gpu", default="0", type=str, help="Assigning GPU id. (-1: use CPU)"
    )
    parser.add_argument("--ckpt", type=str, help="Path to load target pretrain model")

    # get parsing results
    args = parser.parse_args()
    return args


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
    args = get_args()

    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)
        wiki_path = config["dataset"]["small_wiki"]
        config = config["QA"]

    torch_device = get_device(config["device_id"])
    torch.cuda.empty_cache()
    logging.set_verbosity(logging.ERROR)

    qa_model = torch.load(config["save"])

    # evaluation on dev data
    if 'jsonl' in args.data_path:
        test_dataset = FeverDatasetTokenize(wiki_path, args.data_path)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["eval"]["batch_size"],
        num_workers=config["eval"]["n_jobs"],
    )
    report = evaluate(qa_model, test_loader)
    print("[Dev Report]")
    print(report)

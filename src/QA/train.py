import csv
import yaml
import shutil
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split, DataLoader
from transformers import get_linear_schedule_with_warmup, logging

from model import RoBertaClassifier
from dataset import FeverDatasetTokenize
from setting import set_random_seed, get_device

from sklearn.metrics import f1_score, accuracy_score, classification_report

with open("config.yaml", "r") as stream:
    config = yaml.safe_load(stream)
    data = config["dataset"]
    config = config["QA"]

set_random_seed(config["seed"])
torch_device = get_device(config["device_id"])
torch.cuda.empty_cache()
logging.set_verbosity(logging.ERROR)


def train(model, train_loader, val_loader=None, config=config):
    model.train()
    model.to(torch_device)

    optimizer = AdamW(
        model.parameters(),
        lr=eval(config["optimizer"]["Adam"]["learning_rate"]),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["optimizer"]["warmup_steps"],
        num_training_steps=len(train_loader) * config["train"]["epochs"],
    )

    if Path("log/QA").is_file():
        shutil.rmtree("log/QA")
    writer = SummaryWriter("log/QA")

    for epoch in range(config["train"]["epochs"]):
        avg_loss, total_loss = 0, 0
        tqdm_train_loader = tqdm(train_loader)
        for step, batch in enumerate(tqdm_train_loader, 1):
            input_ids = batch["input_ids"].to(torch_device)
            attention_mask = batch["attention_mask"].to(torch_device)
            answer = batch["label"].to(torch_device)

            optimizer.zero_grad()
            preds, loss = model(input_ids, attention_mask, answer)
            loss.backward()
            optimizer.step()
            scheduler.step()

            avg_loss += loss.item()
            total_loss += loss.item()

            if val_loader is not None and step == len(train_loader):
                val_loss, val_acc = evaluate(model, val_loader)
                train_loss = total_loss / len(train_loader)
                tqdm_train_loader.set_description(
                    f"[Epoch:{epoch:03}] Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val F1: {val_acc:.3f}",
                )
                writer.add_scalar("eval/Accuracy", val_acc, epoch)
                writer.add_scalar("eval/Loss", val_loss, epoch)
                writer.add_scalar("train/Loss", train_loss, epoch)

            elif step % config["train"]["log_step"] == 0:
                avg_loss /= config["train"]["log_step"]
                tqdm_train_loader.set_description(
                    f"[Epoch:{epoch}] Train Loss:{avg_loss:.3f}"
                )
                avg_loss = 0

    writer.close()
    return model


@torch.no_grad()
def evaluate(model, val_loader, report=False):
    model.eval()
    model.to(torch_device)

    val_loss = []
    truth, preds = [], []
    for batch in val_loader:
        input_ids = batch["input_ids"].to(torch_device)
        attention_mask = batch["attention_mask"].to(torch_device)
        answer = batch["label"].to(torch_device)

        pred, loss = model(input_ids, attention_mask, answer)
        truth.extend(answer.tolist())
        preds.extend(pred.tolist())
        val_loss.append(loss.item())

    if report:
        return classification_report(truth, preds)

    val_acc = f1_score(truth, preds, average="macro")
    val_loss = np.mean(val_loss)

    model.train()

    return val_loss, val_acc


if __name__ == "__main__":
    train_dataset = FeverDatasetTokenize(data["small_wiki"], data["train_data"])

    val_size = int(len(train_dataset) * config["eval"]["size"])
    train_size = len(train_dataset) - val_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"]["n_jobs"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["eval"]["batch_size"],
        num_workers=config["eval"]["n_jobs"],
    )

    # Training Start
    qa_model = train(RoBertaClassifier(config), train_loader, val_loader)
    torch.save(qa_model, config["save"])

    # Evaluate on dev data
    test_dataset = FeverDatasetTokenize(data["small_wiki"], data["dev_data"])
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["eval"]["batch_size"],
        num_workers=config["eval"]["n_jobs"],
    )
    report = evaluate(qa_model, test_loader, report=True)
    print("[Dev Report]")
    print(report)

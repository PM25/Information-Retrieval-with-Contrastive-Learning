import csv
import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split, DataLoader
from transformers import get_linear_schedule_with_warmup, logging

from dataset import FeverDatasetTokenize
from qa_model import RoBertaClassifier
from setting import set_random_seed, get_device

from sklearn.metrics import f1_score, accuracy_score

with open("config.yaml", "r") as stream:
    configs = yaml.safe_load(stream)
    configs["seed"] = 1009
    configs["device_id"] = 0
    configs["val_size"] = 0.2
    configs["warmup_steps"] = 500
    configs["learning_rate"] = 5e-3
    configs["epochs"] = 15
    configs["log_step"] = 5
    configs["batch_size"] = 6
    configs["hidden_dim"] = 300
    configs["n_cls_layers"] = 3
    configs["freeze_bert"] = False
    configs["dropout"] = 0.2

set_random_seed(configs["seed"])
torch_device = get_device(configs["device_id"])
torch.cuda.empty_cache()
logging.set_verbosity(logging.ERROR)


def train(model, train_loader, val_loader=None, configs=configs):
    model.train()
    model.to(torch_device)

    optimizer = AdamW(
        model.parameters(),
        lr=configs["learning_rate"],
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=configs["warmup_steps"],
        num_training_steps=len(train_loader) * configs["epochs"],
    )

    writer = SummaryWriter("log/QA")

    for epoch in range(configs["epochs"]):
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
                    f"[Epoch:{epoch:03}] Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f}",
                )
                writer.add_scalar("Risk_Accuracy/valalidation", val_acc, epoch)
                writer.add_scalar("Risk_Loss/validation", val_loss, epoch)
                writer.add_scalar("Risk_Loss/train", train_loss, epoch)

            elif step % configs["log_step"] == 0:
                avg_loss /= configs["log_step"]
                tqdm_train_loader.set_description(
                    f"[Epoch:{epoch}] Train Loss:{avg_loss:.3f}"
                )
                avg_loss = 0

    writer.close()
    return model


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    model.to(torch_device)

    val_loss = []
    truth, preds = [], []
    for step, batch in enumerate(val_loader):
        input_ids = batch["input_ids"].to(torch_device)
        attention_mask = batch["attention_mask"].to(torch_device)
        answer = batch["label"].to(torch_device)

        pred, loss = model(input_ids, attention_mask, answer)
        truth.extend(answer.tolist())
        preds.extend(pred.tolist())
        val_loss.append(loss.item())

    val_acc = accuracy_score(truth, preds)
    val_loss = np.mean(val_loss)

    model.train()

    return val_loss, val_acc


def save_preds(model, data_loader):
    model.eval()
    model.to(torch_device)

    all_preds = []
    for step, batch in enumerate(data_loader):
        _id = batch["id"]
        labels = batch["label"]
        input_ids = batch["input_ids"].to(torch_device)
        attention_mask = batch["attention_mask"].to(torch_device)

        preds = model.pred_label(input_ids, attention_mask, labels)
        all_preds.extend(list(zip(_id.tolist(), preds)))

    Path("output").mkdir(parents=True, exist_ok=True)
    with open("output/qa.csv", "w") as f:
        csvwriter = csv.writer(f, delimiter=",")
        csvwriter.writerow(["id", "answer"])
        for _id, pred in all_preds:
            csvwriter.writerow([_id, pred])
    with open("output/qa_configs.yml", "w") as yaml_file:
        yaml.dump(configs, yaml_file, default_flow_style=False)
    print("*Successfully saved prediction to output/qa.csv")


if __name__ == "__main__":
    dataset = FeverDatasetTokenize(
        configs["dataset"]["small_wiki"], configs["dataset"]["data_dir"]
    )

    val_size = int(len(dataset) * configs["val_size"])
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=configs["batch_size"], shuffle=True, num_workers=16
    )
    val_loader = DataLoader(
        val_dataset, batch_size=configs["batch_size"], num_workers=16
    )

    qa_model = train(RoBertaClassifier(configs), train_loader, val_loader)

    # test_dataset = dataset(configs, configs["dev_qa_data"])
    # test_loader = DataLoader(
    #     test_dataset, batch_size=configs["batch_size"], num_workers=4
    # )
    # save_preds(qa_model, test_loader)
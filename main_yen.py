#%%
import yaml
from dataset import FeverDataset

with open("config.yaml", "r") as stream:
    config = yaml.safe_load(stream)

dataset = FeverDataset(config["small_wiki"], config["data_dir"])
print(dataset[1])
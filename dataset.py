#%%
import json
import yaml
from tqdm import tqdm
from pathlib import Path
from unicodedata import normalize

from torch.utils.data import Dataset

with open('config.yaml', 'r') as stream:
    config = yaml.safe_load(stream)


def process_wiki():    
    with open(config['wiki_out'], 'r') as f:
        wiki = json.load(f)    
    
    for datum in tqdm(wiki.values()):
        lines = []
        for line in datum["lines"].split("\n"):
            line = line.split('\t')
            line = ' '.join(line[1:])
            lines.append(line)
        
        datum['lines'] = lines
    
    return wiki


def process_trainjsonl():
    with open(config['train_data'], "r", encoding="utf-8") as f:
        json_strs = f.readlines()
    
    out = []

    for json_str in json_strs:
        dic = json.loads(json_str)
        claim = dic['claim']
        label = dic['label']
        
        clean_evidences = set()
        for evidences in dic["evidence"]:
            for evidence in evidences:
                if(evidence[2] is not None):
                    doc_id = normalize("NFKD", evidence[2])
                    sent_id = evidence[3]
                    clean_evidences.add((doc_id, sent_id))
        
        out.append(
            {
                "claim": claim,
                "label": label,
                "evidences": clean_evidences,
            }
        )
    
    return out



class FeverDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.wiki = process_wiki()
        
        train_data = process_trainjsonl()
        self.process(train_data)
        
        self.label_map = {"SUPPORTS": 0, "REFUTES": 1}
        
        self.data = []
        for datum in train_data:
            label = datum["label"]  # SUPPORTS, NOT VERIFIABLE, REFUTES
            if(label == "NOT ENOUGH INFO"):
                continue
            
            self.data.append(
                {
                    "label": self.label_map[label],
                    "claim": datum["claim"],
                    "evidences": datum["evidences"],
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def process(self, data):
        for datum in data:
            evidences = datum["evidences"]
            process_evidences = []
            for doc_id, sent_id in evidences:
                sent = self.wiki[doc_id]['lines'][sent_id]
                process_evidences.append(
                    {
                        "title": doc_id,
                        "sentence": sent
                    }
                )
            datum["evidences"] = process_evidences

if __name__ == "__main__":
    dataset = FeverDataset()
    # print(dataset[0])
#%%
import json
import yaml
from tqdm import tqdm
from pathlib import Path
from unicodedata import normalize

from torch.utils.data import Dataset

with open('config.yaml', 'r') as stream:
    config = yaml.safe_load(stream)


def process_wiki(fname):    
    with open(fname, 'r') as f:
        wiki = json.load(f)    
    
    for datum in tqdm(wiki.values()):
        lines = []
        for line in datum["lines"].split("\n"):
            line = line.split('\t')
            line = ' '.join(line[1:])
            lines.append(line)
        
        datum['lines'] = lines
    
    return wiki


def process_trainjsonl(fname):
    with open(fname, "r", encoding="utf-8") as f:
        json_strs = f.readlines()
    
    out = []

    for json_str in json_strs:
        dic = json.loads(json_str)
        _id = dic['id']
        claim = dic['claim']
        label = dic['label']
        
        # One sample of clean_evidence
        # {doc_id: [sent_id, sent_id, ...]}
        clean_evidences = {}
        for evidences in dic["evidence"]:
            for evidence in evidences:
                if(evidence[2] is not None):
                    doc_id = normalize("NFKD", evidence[2])
                    sent_id = evidence[3]
                    clean_evidences[doc_id] = clean_evidences.get(doc_id, []) + [sent_id]
        
        out.append(
            {
                "id": _id, 
                "claim": claim,
                "label": label,
                "evidences": clean_evidences,
            }
        )
    
    return out



class FeverDataset(Dataset):
    def __init__(self, wiki_path=config['wiki_out'], train_path=config['train_data']):
        super().__init__()
        self.wiki = process_wiki(wiki_path)
        
        train_data = process_trainjsonl(train_path)
        train_data = self.process(train_data)
        
        self.label_map = {"SUPPORTS": 1, "REFUTES": 0}
        
        self.data = []
        for datum in train_data:
            label = datum["label"]  # SUPPORTS, NOT VERIFIABLE, REFUTES
            if(label == "NOT ENOUGH INFO"):
                continue
            
            self.data.append(
                {
                    "id": datum['id'],
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
        data = data.copy()
        for datum in data:
            evidences = datum["evidences"]
            process_evidences = []
            
            for doc_id, sent_ids in evidences.items():
                process_evidences.append(
                    {
                        "title": doc_id,
                        "document": self.wiki[doc_id]['lines'],
                        "sent_idx": sent_ids
                    }
                )
            datum["evidences"] = process_evidences
        
        return data

if __name__ == "__main__":
    dataset = FeverDataset()
    print(dataset[0])
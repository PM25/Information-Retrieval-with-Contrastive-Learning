#%%
import ray
import json
from pathlib import Path
from unicodedata import normalize

from torch.utils.data import Dataset
from transformers import RobertaTokenizer


class TorchDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        preprocessdata = PreprocessData(data_dir)
        trainjsonl = preprocessdata.get_trainjsonl()
        
        self.labels = {"SUPPORTS": 0, "REFUTES": 1}
        
        self.data = []
        for trainjson in trainjsonl:
            claim = trainjson["claim"]
            evidences = trainjson["evidence_sentences"]
            label = trainjson["label"]  # SUPPORTS, NOT VERIFIABLE, REFUTES
            if(label == "NOT ENOUGH INFO"):
                continue
            self.data.append(
                {
                    "label": self.labels[trainjson["label"]],
                    "claim": trainjson["claim"],
                    "evidences": evidences,
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PreprocessData:
    def __init__(self, data_dir="data"):
        ray.init()
        self.wikipages_dir = Path(data_dir) / "wiki-pages"
        self.trainjsonl_pth = Path(data_dir) / "train.jsonl"
        self.shared_task_dev_pth = Path(data_dir) / "train_task_dev.jsonl"
        self.wikipages = None
        self.trainjsonl = None

    def get_wikipages(self):
        if self.wikipages is None:
            self.wikipages = self.load_wikipages()
        
        return self.wikipages

    def get_trainjsonl(self):
        if self.trainjsonl is None:
            self.trainjsonl = self.load_trainjsonl()
        
        return self.trainjsonl

    #%% load wiki-pages
    def load_wikipages(self):
        fnames = list(self.wikipages_dir.glob("wiki-*.jsonl"))
        wiki_data_dicts = [load_single_wikipages.remote(fname) for fname in fnames]
        
        data = {}
        for wiki_data_dict in ray.get(wiki_data_dicts):
            data.update(wiki_data_dict)
        
        return data

    #%% load train_jsonl
    def load_trainjsonl(self):
        wikipages = self.get_wikipages()
        
        with open(self.trainjsonl_pth, "r", encoding="utf-8") as f:
            json_strs = f.readlines()
        
        json_list = [json.loads(json_str) for json_str in json_strs]
        
        for dic in json_list:
            processed_evidence_sets = []
            
            for evidence_sets in dic["evidence"]:
                evidence_sents = []
                
                a = [(e[2], e[3]) for e in evidence_sets]
                dupes = [x for n, x in enumerate(a) if x in a[:n]]
                print(dic["claim"])
                print(dupes)

                for evidence in evidence_sets:
                    if(evidence[2] is None or evidence[3] is None):
                        continue
                    
                    evidence_id = normalize("NFKD", evidence[2])
                    sent_id = evidence[3]
                    
                    print('len >', len(wikipages[evidence_id]["lines"]), sent_id)
                    evidence_sent = wikipages[evidence_id]["lines"][sent_id]
                    print(evidence_sent)
                    evidence_sents.append(evidence_sent)
                
                processed_evidence_sets.append(evidence_sents)
            
            dic["evidence_sentences"] = processed_evidence_sets
        
        return json_list


@ray.remote
def load_single_wikipages(fname):
    print(f"Processing {fname}")
    
    with open(fname, "r", encoding="utf-8") as f:
        json_strs = f.readlines()
        
    json_list = [json.loads(json_str) for json_str in json_strs]

    wiki_data = to_dict(json_list)
    for datum in wiki_data.values():
        lines = []
        for line in datum["lines"].split("\n"):
            line = line.split('\t')
            line = ' '.join(line[1:])
            lines.append(line)
        
        datum['lines'] = lines
    
    return wiki_data


#%% turn list of dictionary (with key id) into a dictionary.
def to_dict(list_of_dict):
    output = {}
    for dic in list_of_dict:
        _id = normalize("NFKD", dic["id"])
        dic.pop("id", None)
        output[_id] = dic

    return output

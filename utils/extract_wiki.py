import ray
import yaml
import json
from tqdm import tqdm
from pathlib import Path
from unicodedata import normalize

ray.init()

with open("config.yaml", "r") as stream:
    config = yaml.safe_load(stream)

def load_wikipages():
    wiki_path = Path(config['wiki_data'])
    fnames = list(wiki_path.glob("wiki-*.jsonl"))
    
    wiki_data_dicts = [load_single_wikipages.remote(fname) for fname in fnames]
    for x in tqdm(to_iterator(wiki_data_dicts), total=len(wiki_data_dicts)):
        pass
    
    data = {}
    for wiki_data_dict in tqdm(ray.get(wiki_data_dicts)):
        data.update(wiki_data_dict)
    
    return data


def to_iterator(obj_ids):
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])


@ray.remote
def load_single_wikipages(fname):
    with open(fname, "r", encoding="utf-8") as f:
        json_strs = f.readlines()
        
    json_list = [json.loads(json_str) for json_str in json_strs]
    
    return to_dict(json_list)


def to_dict(list_of_dict):
    output = {}
    for dic in list_of_dict:
        _id = normalize("NFKD", dic["id"])
        dic.pop("id", None)
        output[_id] = dic

    return output

def trainjsonl_documents():
    with open(config['train_data'], "r", encoding="utf-8") as f:
        json_strs = f.readlines()

    documents = set()
    for json_str in json_strs:
        dic = json.loads(json_str)
        for evidence_sets in dic["evidence"]:
            for evidence in evidence_sets:
                if evidence[2] is not None:
                    documents.add(evidence[2])
    
    return documents


if __name__ == "__main__":
    wikipages = load_wikipages()
    documents = trainjsonl_documents()

    wiki = {doc: wikipages[doc] for doc in documents}

    out_path = Path(config['wiki_out'])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w', encoding='utf8') as f:
        json.dump(wiki, f, indent=4, ensure_ascii=False)
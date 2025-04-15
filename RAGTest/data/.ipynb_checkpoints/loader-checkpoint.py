import os
import json
from llama_index.core import Document
from config import Config

cfg = Config()


def get_documents():
    documents = []
    with open("/root/autodl-tmp/zh/ragx_old/data/B.json", 'r', encoding='utf-8') as file:
        data = json.load(file)
        
    # for _, entry in data.items():
    #     for _, passage in entry.items():
    #         title = ""
    #         text = "" + passage['page_content']
    #         id = passage['index']
    #         ducument = Document(text=text, metadata={'title': title, 'id': id}, doc_id=str(id))
    #         documents.append(ducument)
    # print("len(B):", len(documents))
    with open("/root/autodl-tmp/zh/ragx_old/data/data_50.json", 'r', encoding='utf-8') as file:
        data = json.load(file)
        for entry in data:
            title = entry["other_info"]["doc_name"]
            for reference, ids in zip(entry["key_content"]["reference"], entry["key_content"]["reference_idx"]):
                text = reference
                id = ids
                ducument = Document(text=text, metadata={'title': title, 'id': id}, doc_id=str(id))
                documents.append(ducument)
    return documents
    # dirs = os.listdir(path)
    # documents = []
    # for dir in dirs:
    #     files = os.listdir(os.path.join(path,dir))
    #     for file in files:
    #         # read json file
    #         with open(os.path.join(path,dir,file),'r',encoding='utf-8') as f:
    #             for line in f.readlines():
    #                 raw = json.loads(line)
    #                 ducument = Document(text=raw['text'],metadata={'title':raw['title']},doc_id=raw['id'])
    #                 documents.append(ducument)
    # return documents
    title2sentenses = sources['title2sentences']
    title2id = sources['title2id']
    documents = [Document(text=' '.join(sentence_list), metadata={'title': title, 'id': title2id[title]},
                          doc_id=str(title2id[title])) for title, sentence_list in title2sentenses.items()]
    if cfg.experiment_1:
        documents = documents[:cfg.test_all_number_documents]
    return documents


if __name__ == '__main__':
    documents = get_documents('../wiki')
    print(documents)

import os
import json
from llama_index.core import Document

def get_documents():
    documents = []
    with open("data/test_corpus.json", 'r', encoding='utf-8') as file:
        data = json.load(file)
        
    for _, entry in data.items():
        for _, passage in entry.items():
            title = ""
            text = "" + passage['page_content']
            id = passage['index']
            ducument = Document(text=text, metadata={'title': title, 'id': id}, doc_id=str(id))
            documents.append(ducument)
            if len(documents) == 6066:
                break
        if len(documents) == 6066:
            break
    print("len(B):", len(documents))
    # with open("data/data_100.json", 'r', encoding='utf-8') as file:
    with open("data/data_50.json", 'r', encoding='utf-8') as file:
        data = json.load(file)
        for entry in data:
            title = entry["other_info"]["doc_name"]
            for reference, ids in zip(entry["key_content"]["reference"], entry["key_content"]["reference_idx"]):
                text = reference
                id = ids
                ducument = Document(text=text, metadata={'title': title, 'id': id}, doc_id=str(id))
                documents.append(ducument)
    return documents

if __name__ == '__main__':
    documents = get_documents('../wiki')
    print(documents)

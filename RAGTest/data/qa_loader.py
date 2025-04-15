import json
import os
import re

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset


def build_split(answers, questions, supporting_facts, title2id, title2sentences):
    golden_ids = []
    golden_sentences = []
    filter_questions = []
    filter_answers = []
    i = 0
    for sup, q, a in zip(supporting_facts, questions, answers):
        # i = i + 1
        # if i == 300:
        #     break
        # if len(sup['sent_id']) == 0:
        #     continue
        try:
            sup_title = sup['title']
            # send_id = sup['sent_id']
            # golden_id = [title2start[t]+i for i,t in zip(send_id,sup_title)]
            sup_titles = set(sup_title)
            golden_id = [title2id[t] for t in sup_titles]


        except:
            continue
        golden_ids.append(golden_id)
        golden_sentences.append([' '.join(title2sentences[t]) for t in sup_titles])
        filter_questions.append(q)
        filter_answers.append(a)
    print("questions:", len(questions))
    print("filter_questions:", len(filter_questions))
    return filter_questions,filter_answers, golden_ids, golden_sentences

def extract_law_name(input_str):
    # 使用正则表达式匹配第一个 '-' 字符后的非汉字字符串
    match = re.search(r'-(.*?)[^\u4e00-\u9fa5]', input_str)

    if match:
        law_name = match.group(1).strip('-')
        return law_name
    else:
        return None


def get_qa_dataset(dataset_name: str):
    if dataset_name == "hotpot_qa":
        dataset = load_dataset("hotpot_qa", "fullwiki")
        questions = dataset['train']['question'] + dataset['test']['question'] + dataset['validation']['question']
        answers = dataset['train']['answer'] + dataset['test']['answer'] + dataset['validation']['answer']
        golden_sources = dataset['train']['context'] + dataset['test']['context'] + dataset['validation']['context']
        supporting_facts = dataset['train']['supporting_facts'] + dataset['test']['supporting_facts'] + \
                           dataset['validation']['supporting_facts']
        source_sentences = []
        title2sentences = {}
        titles = []
        title2start = {}
        title2id = {}
        id = 0
        cur = 0
        i = 0
        for sup, source in zip(supporting_facts, golden_sources):
            i = i + 1
            if i == 300:
                break
            title = source['title']
            sentence = source['sentences']
            for t, s in zip(title, sentence):
                if t not in title2sentences:
                    title2sentences[t] = s
                    title2start[t] = cur
                    titles.append(t)
                    source_sentences.extend(s)
                    cur += len(s)
                    title2id[t] = id
                    id += 1
                else:
                    print("title already exists, skip.")

        golden_ids = []
        golden_sentences = []
        filter_questions = []
        filter_answers = []
        i = 0

        for sup, q, a in zip(supporting_facts, questions, answers):
            i = i + 1
            if i == 300:
                break
            if len(sup['sent_id']) == 0:
                continue
            try:
                sup_title = sup['title']
                send_id = sup['sent_id']
                # golden_id = [title2start[t]+i for i,t in zip(send_id,sup_title)]
                sup_titles = set(sup_title)
                golden_id = [title2id[t] for t in sup_titles]


            except:
                continue
            golden_ids.append(golden_id)
            golden_sentences.append([' '.join(title2sentences[t]) for t in sup_titles])
            filter_questions.append(q)
            filter_answers.append(a)
        print("questions:", len(questions))
        print("filter_questions:", len(filter_questions))
        return dict(
            question=filter_questions,
            answers=filter_answers,
            golden_ids=golden_ids,
            golden_sentences=golden_sentences,
            sources=source_sentences,
            titles=titles,
            title2sentences=title2sentences,
            title2start=title2start,
            title2id=title2id,
            dataset=dataset)


    elif dataset_name == "json_download":
        # with open("/root/autodl-tmp/zh/ragx_old/data/data_100.json", 'r', encoding='utf-8') as file:
        with open("/root/autodl-tmp/zh/ragx_old/data/data_50.json", 'r', encoding='utf-8') as file:
            data = json.load(file)
        questions = []
        answers = []
        golden_sources = []
        golden_ids = []
        question_types = []
        text2id = {}
        dataset = data
        for entry in data:
            question_types.append(entry["other_info"]["question_type"])
            entry = entry["key_content"]
            question = entry['question']
            answer = entry['answer']
            references = entry['reference']
            ids = entry["reference_idx"]
            if answer=="":
                continue
            questions.append(question)
            answers.append(answer)
            golden_sources.append(references)
            golden_ids.append(ids)
        
    #     for entry in data:
    #         title = entry["other_info"]["doc_name"]
    #         for reference, ids in zip(entry["key_content"]["reference"], entry["key_content"]["reference_idx"]):
    #             text = reference
    #             id = ids
    #             ducument = Document(text=text, metadata={'title': title, 'id': id}, doc_id=str(id))
    #             documents.append(ducument)
        
    #     with open("/root/autodl-tmp/zh/ragx_old/data/qa.json", 'r', encoding='utf-8') as file:
    #         data = json.load(file)

    #     questions = []
    #     answers = []
    #     golden_sources = []
    #     golden_ids = []
    #     text2id = {}

    #     dataset = data

    #     for entry in data:
    #         question = entry['question']
    #         answer = entry['answer']
    #         references = entry['reference']
    #         ids = entry['ids']
    #         if answer=="":
    #             continue
    #         questions.append(question)
    #         answers.append(answer)
    #         golden_sources.append(references)
    #         golden_ids.append(ids)

    # else:
    #     raise NotImplementedError(f'dataset {dataset_name} not implemented!')

    return dict(
        question=questions,
        answers=answers,
        golden_sentences=golden_sources,
        golden_ids=golden_ids,
        question_types=question_types,
        dataset=dataset,
        title2id=text2id)


if __name__ == '__main__':
    name = 'json_download'  # drop, natural_questions, hotpot_qa
    data = get_qa_dataset(name)


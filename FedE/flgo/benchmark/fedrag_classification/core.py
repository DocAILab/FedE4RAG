import urllib
import zipfile
import torch
from torch.utils.data import TensorDataset, Dataset
from transformers import BertTokenizer

from flgo.benchmark.toolkits import BasicTaskGenerator, BasicTaskCalculator
from flgo.benchmark.base import BasicTaskPipe
import collections
import re
import os
import os.path

from flgo.benchmark.toolkits.nlp.classification import GeneralCalculator

try:
    import ujson as json
except:
    import json
import flgo.benchmark
import os.path
import torch

from torch import nn


def collate_batch(batch):
    label_list, text_list = [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(_text, dtype=torch.int64)
        text_list.append(processed_text)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    return text_list, label_list


def cos_sim(a, b):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    if len(a.shape) == 1:
        a = a.squeeze(0)
    if len(b.shape) == 1:
        b = b.squeeze(0)
    a_norm = torch.nn.functional.normalize(a, dim=-1, p=2)
    b_norm = torch.nn.functional.normalize(b, dim=-1, p=2)
    return torch.mm(a_norm, torch.transpose(b_norm, 0, 1))


class FEDRAG(Dataset):
    def __init__(self, train=True):
        self.train = train
        # TODO 加载数据集
        with open("./select_data.json", 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.questions = []
        self.id = []
        self.ref = []

        for entry in data:
            question = entry['question']
            answer = entry['company']
            merged_text = entry['reference']
            self.questions.append(question)
            self.id.append(answer)
            self.ref.append(merged_text)

    def __getitem__(self, index):
        return self.questions[index], self.id[index], self.ref[index]

    def __len__(self):
        return len(self.questions)


class TaskGenerator(BasicTaskGenerator):
    # TODO 加载数据集
    def __init__(self, rawdata_path="./select_data.json"):
        super(TaskGenerator, self).__init__(benchmark='fedrag_classification', rawdata_path=rawdata_path)
        # Regular expression to capture an actors name, and line continuation

    def load_data(self):
        self.train_data = FEDRAG(train=True)
        self.test_data = FEDRAG(train=False)
        return

    def partition(self):
        self.local_datas = self.partitioner(self.train_data)


class TaskPipe(BasicTaskPipe):
    class TaskDataset(torch.utils.data.Subset):
        def __init__(self, dataset, indices):
            super().__init__(dataset, indices)
            self.dataset = dataset
            self.indices = indices

        def __getitem__(self, idx):
            if isinstance(idx, list):
                return self.dataset[[self.indices[i] for i in idx]]
            return self.dataset[self.indices[idx]]

    def save_task(self, generator):
        client_names = self.gen_client_names(len(generator.local_datas))
        feddata = {'client_names': client_names, 'server_data': list(range(len(generator.test_data))),
                   'rawdata_path': generator.rawdata_path}
        for cid in range(len(client_names)): feddata[client_names[cid]] = {'data': generator.local_datas[cid], }
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)
        return

    def load_data(self, running_time_option) -> dict:
        # load the datasets
        train_data = FEDRAG(train=True)
        test_data = FEDRAG(train=False)
        # rearrange data for server
        server_data_test, server_data_val = self.split_dataset(test_data, running_time_option['test_holdout'])
        task_data = {'server': {'test': server_data_test, 'val': server_data_val}}
        # rearrange data for clients
        for cid, cname in enumerate(self.feddata['client_names']):
            cdata = self.TaskDataset(train_data, self.feddata[cname]['data'])
            cdata_train, cdata_val = self.split_dataset(cdata, running_time_option['train_holdout'])
            task_data[cname] = {'train': cdata_train, 'val': cdata_val}
        return task_data


class TaskCalculator(GeneralCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(GeneralCalculator, self).__init__(device, optimizer_name)
        self.DataLoader = torch.utils.data.DataLoader
        self.criterion = torch.nn.CrossEntropyLoss()
        # TODO 加载模型
        self.tokenizer = BertTokenizer.from_pretrained('/path/to/bge-en')

    def compute_client_loss(self, server_logits, model, batch_data):
        questions = batch_data[0]
        answers = batch_data[1]
        references = batch_data[2]

        max_length = self.tokenizer.model_max_length

        question_inputs = self.tokenizer(questions, return_tensors="pt", padding=True, truncation=True,
                                         max_length=max_length)
        question_inputs.to(self.device)
        question_outputs = model(**question_inputs)
        question_pooled_tensors = torch.mean(question_outputs.last_hidden_state, dim=1, keepdim=False)

        reference_inputs = self.tokenizer(references, return_tensors="pt", padding=True, truncation=True,
                                          max_length=max_length)
        reference_inputs.to(self.device)
        reference_outputs = model(**reference_inputs)
        reference_pooled_tensors = torch.mean(reference_outputs.last_hidden_state, dim=1, keepdim=False)

        logits = cos_sim(question_pooled_tensors, reference_pooled_tensors)
        logits.to(self.device)

        label = torch.arange(len(logits)).to(self.device)
        criterion = nn.CrossEntropyLoss()
        loss_1 = criterion(logits, label)

        criterion2 = nn.MSELoss()
        loss_2 = criterion2(logits, server_logits)
        return loss_1 + 100 * loss_2, loss_1, loss_2

    def compute_server_loss(self, model, batch_data):
        questions = batch_data[0]
        answers = batch_data[1]
        references = batch_data[2]

        max_length = self.tokenizer.model_max_length

        question_inputs = self.tokenizer(questions, return_tensors="pt", padding=True, truncation=True,
                                         max_length=max_length)
        question_inputs.to(self.device)
        with torch.no_grad():
            question_outputs = model(**question_inputs)
        question_pooled_tensors = torch.mean(question_outputs.last_hidden_state, dim=1, keepdim=False)

        reference_inputs = self.tokenizer(references, return_tensors="pt", padding=True, truncation=True,
                                          max_length=max_length)
        reference_inputs.to(self.device)
        with torch.no_grad():
            reference_outputs = model(**reference_inputs)
        reference_pooled_tensors = torch.mean(reference_outputs.last_hidden_state, dim=1, keepdim=False)

        logits = cos_sim(question_pooled_tensors, reference_pooled_tensors)
        return logits

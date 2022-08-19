import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np
from transformers import AutoTokenizer
import pandas as pd
import os

PATH = '/content/drive/MyDrive/NLP'
train = pd.read_csv(os.path.join(PATH, 'train_data.csv'), encoding='utf-8')
checkpoint = 'klue/roberta-large'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def label_to_num(label):
    label_dict = {"entailment": 0, "contradiction": 1, "neutral": 2, "answer": 3}
    num_label = []
    for v in label:
        num_label.append(label_dict[v])
    return num_label

class BERTDataset(torch.utils.data.Dataset):
    def __init__(self, pair_dataset, label):
        self.pair_dataset = pair_dataset
        self.label = label
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item['labels'] = torch.tensor(self.label[idx])
        return item
    def __len__(self):
        return len(self.label)

def data_loader():
  tokenized_train = tokenizer(list(train['premise']), list(train['hypothesis']),
    return_tensors="pt",
    max_length=128,
    padding=True,
    truncation=True,
    add_special_tokens=True
    )

  kfold = StratifiedKFold(n_splits=5, shuffle=True)
  for train_ids, test_ids in kfold.split(train, train['label']):
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

  train_label = label_to_num(train['label'].values)
  train_dataset = BERTDataset(tokenized_train, train_label)

  train_dataloader = torch.utils.data.DataLoader(
                      train_dataset, 
                      batch_size=16, sampler=train_subsampler)
  eval_dataloader = torch.utils.data.DataLoader(
                      train_dataset,
                      batch_size=16, sampler=test_subsampler)
  return train_dataloader, eval_dataloader
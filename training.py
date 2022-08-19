import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np
from transformers import AutoTokenizer
import pandas as pd
import os
from accelerate import Accelerator
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from transformers import AutoModel,AutoModelForSequenceClassification, AutoConfig
from transformers import AdamW
from tqdm import tqdm
from tqdm.auto import tqdm
import gc

PATH = '/content/drive/MyDrive/NLP'
train = pd.read_csv(os.path.join(PATH, 'train_data.csv'), encoding='utf-8')
test = pd.read_csv(os.path.join(PATH, 'test_data.csv'), encoding='utf-8')
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

train_dataloader, eval_dataloader = data_loader()
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

def training_function(train_dataloader=train_dataloader, eval_dataloader=eval_dataloader):
  accelerator = Accelerator()
  for fold in range(5):
    config = AutoConfig.from_pretrained(checkpoint)
    config.num_labels = 3
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, config=config)

    optimizer = AdamW(model.parameters(), lr=	1e-5)

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)

    num_epochs = 10
    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))
    lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer=optimizer,  
        num_warmup_steps=1,
        num_training_steps=num_training_steps,
    )

    for epoch in range(num_epochs):
        train_acc = 0.0
        test_acc = 0.0

        model.train()
        for batch_id, batch in enumerate(train_dataloader):
            outputs = model(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])
            loss = F.cross_entropy(outputs[0], batch['labels'])
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            train_acc += calc_accuracy(outputs.logits, batch['labels'])
        print("epoch {} train acc {}".format(epoch+1, train_acc / (batch_id+1)))

        model.eval()
        for batch_id, batch in enumerate(eval_dataloader):
          with torch.no_grad():
              outputs = model(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])

          test_acc += calc_accuracy(outputs.logits, batch['labels'])
        print("epoch {} test acc {}".format(epoch+1, test_acc / (batch_id+1)))
        gc.collect()
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained('/content/drive/MyDrive/220228/model' + str(fold), save_function=accelerator.save)
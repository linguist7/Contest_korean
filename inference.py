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
test = pd.read_csv(os.path.join(PATH, 'test_data.csv'), encoding='utf-8')
checkpoint = 'klue/roberta-large'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def label_to_num(label):
    label_dict = {"entailment": 0, "contradiction": 1, "neutral": 2, "answer": 3}
    num_label = []
    for v in label:
        num_label.append(label_dict[v])
    return num_label

def num_to_label(label):  
  label_dict = {0: "entailment", 1: "contradiction", 2: "neutral"}
  str_label = []
  for i, v in enumerate(label):
      str_label.append([i,label_dict[v]])
  return str_label

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
  tokenized_test = tokenizer(
      list(test['premise']),
      list(test['hypothesis']),
      return_tensors="pt",
      max_length=128,
      padding=True,
      truncation=True,
      add_special_tokens=True
  )
  test_label = label_to_num(test['label'].values)
  test_dataset = BERTDataset(tokenized_test, test_label)
  test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

  return test_dataloader

test_dataloader = data_loader()
def inference():
  for fold in range(5):

    config = AutoConfig.from_pretrained(checkpoint)
    config.num_labels = 3
    model = AutoModelForSequenceClassification.from_pretrained('/content/drive/MyDrive/220228/model' + str(fold), num_labels=3)
    model.resize_token_embeddings(tokenizer.vocab_size)
    accelerator = Accelerator()
    model = accelerator.unwrap_model(model)

    output_pred = []
    output_prob = []

    model, dataloader= accelerator.prepare(model, dataloader)

    model.eval()

    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs = model(
                input_ids=data['input_ids'],
                attention_mask=data['attention_mask']
            )
        logits = outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)
        output_pred.append(result)
        output_prob.append(prob)
        
    pred_answer, output_prob = np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

    answer = num_to_label(pred_answer)

    df_label = pd.DataFrame(answer, columns=['index', 'label'])
    df_prob = pd.DataFrame(output_prob)

    df_label.to_csv('/content/drive/MyDrive/220228/pred_label'+str(fold)+'.csv', index=False)
    df_prob.to_csv('/content/drive/MyDrive/220228/pred_prob'+str(fold)+'.csv', index=False)
  pred0 = pd.read_csv('/content/drive/MyDrive/sm_220228/pred_prob0.csv')
  pred1 = pd.read_csv('/content/drive/MyDrive/sm_220228/pred_prob1.csv')
  pred2 = pd.read_csv('/content/drive/MyDrive/sm_220228/pred_prob2.csv')
  pred3 = pd.read_csv('/content/drive/MyDrive/sm_220228/pred_prob3.csv')
  pred4 = pd.read_csv('/content/drive/MyDrive/sm_220228/pred_prob4.csv')
  pred = pd.DataFrame((np.array(pred0) + np.array(pred1) + np.array(pred2) + np.array(pred3) + np.array(pred4))/5)
  test = pd.read_csv(os.path.join(PATH, 'test_data.csv'), encoding='utf-8')
  test = pd.concat([test, pred], axis=1)
  answer = num_to_label(np.argmax(np.array(pred), axis=-1))
  test['label'] = pd.DataFrame(answer)[1]
  return test
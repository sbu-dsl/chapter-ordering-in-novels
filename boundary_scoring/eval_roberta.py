import csv
import torch
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments

rand_seed = 1729
torch.manual_seed(rand_seed)
np.random.seed(rand_seed)

class ChapterBoundaryDataset(torch.utils.data.Dataset):
    def __init__(self, gids, chapter_orders, encodings, labels):
        self.gids = gids
        self.chapter_orders = chapter_orders
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = self.encodings[idx]
        item['labels'] = torch.tensor(self.labels[idx])
        item['gid'] = self.gids[idx]
        item['chapter_order'] = self.chapter_orders[idx]
        return item

    def __len__(self):
        return len(self.labels)

print("Loading test dataset")
test_dataset = torch.load("test_dataset")
# print(len(test_dataset))

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    learning_rate=1e-5,
    logging_dir='./logs',            # directory for storing logs
    save_steps=1000,
    logging_steps=10,
)

model = RobertaForSequenceClassification.from_pretrained('chapter_order_model')

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
)

def compute_metrics(preds, labels):
    preds = preds.argmax(-1)
    print(preds.shape)
    print(labels.shape)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

results = trainer.predict(test_dataset)
torch.save(results, 'results.pt')
# results = torch.load('results.pt')
logits = results.predictions
print(logits.shape)
with open("test_chapter_orders.csv", "w") as csvfile:
    headers = ["gid", "left_chapter", "right_chapter", "left_score", "right_score"]
    writer = csv.DictWriter(csvfile, headers)
    writer.writeheader()
    for idx, d in tqdm(enumerate(test_dataset)):
        gid = d['gid']
        left, right = d['chapter_order']
        leftlogit, rightlogit = logits[idx]
        writer.writerow({
            "gid": gid,
            "left_chapter": left,
            "right_chapter": right,
            "left_score": leftlogit,
            "right_score": rightlogit
        })

print(compute_metrics(results.predictions, results.label_ids))

import torch
import numpy as np
from pathlib import Path
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments

rand_seed = 1729
torch.manual_seed(rand_seed)
np.random.seed(rand_seed)

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

class ChapterBoundaryDataset(torch.utils.data.Dataset):
    def __init__(self, gids, chapter, encodings, labels):
        self.gids = gids
        self.chapter = chapter
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = self.encodings[idx]
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_datasets(save_name):
    if Path(save_name).exists():
        return torch.load(save_name)

with open("train_gids.txt") as f:
    train_gids = [x.strip() for x in f.readlines()]
with open("eval_gids.txt") as f:
    val_gids = [x.strip() for x in f.readlines()]
with open("test_gids.txt") as f:
    test_gids = [x.strip() for x in f.readlines()]
print(len(train_gids), len(val_gids), len(test_gids))

train_dataset = load_datasets("train_dataset")
val_dataset = load_datasets("val_dataset")

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=10,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    learning_rate=1e-5,
    logging_dir='./logs',            # directory for storing logs
    save_steps=1000,
    logging_steps=10,
)

model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()
trainer.save_model('first_chapter_model')
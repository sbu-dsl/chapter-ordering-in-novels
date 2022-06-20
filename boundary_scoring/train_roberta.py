import torch
import os
import pickle
import numpy as np
import itertools
from tqdm import tqdm
from pathlib import Path
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments

rand_seed = 1729
torch.manual_seed(rand_seed)
np.random.seed(rand_seed)

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

class ChapterBoundaryDataset(torch.utils.data.Dataset):
    def __init__(self, gids, chapter_orders, encodings, labels):
        self.gids = gids
        self.chapter_orders = chapter_orders
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = self.encodings[idx]
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def generate_examples(pkl_path):
    MAX_WINDOW_TOKEN_LENGTH = 512
    # 4 special tokens (1 at start, 1 at end, 2 between chapters)
    MAX_CHAPTER_TOKEN_LENGTH = (MAX_WINDOW_TOKEN_LENGTH - 4) // 2
    with open(pkl_path, "rb") as f:
        all_tokens = pickle.load(f)
    
    chapter_starts = []
    chapter_ends = []
    for tokens in all_tokens:
        start = []
        for toks in tokens:
            if len(start) + len(toks) <= MAX_CHAPTER_TOKEN_LENGTH:
                start.extend(toks)
            else:
                break
        chapter_starts.append(start)

        end = []
        total_toks = 0
        for toks in reversed(tokens):
            if total_toks + len(toks) <= MAX_CHAPTER_TOKEN_LENGTH:
                end.append(toks)
                total_toks += len(toks)
            else:
                break
        end.reverse()
        end= list(itertools.chain.from_iterable(end))
        chapter_ends.append(end)

    chapter_orders = []
    encodings = []
    labels = []
    num_chapters = len(all_tokens)
    for i in range(num_chapters):
        for j in range(i+1, num_chapters):
            # end of chapter i + start of chapter j
            left = chapter_ends[i]
            right = chapter_starts[j]
            chapter_orders.append((i,j))
            encoding = tokenizer.prepare_for_model(left, right, padding='max_length', return_tensors='pt')
            encodings.append(encoding)
            if j == i+1:
                labels.append(1)
            else:
                labels.append(0)
            
            
            # end of chapter j + start of chapter i
            left = chapter_ends[j]
            right = chapter_starts[i]
            chapter_orders.append((j,i))
            encoding = tokenizer.prepare_for_model(left, right, padding='max_length', return_tensors='pt')
            encodings.append(encoding)
            labels.append(0)
    return chapter_orders, encodings, labels

def create_datasets(all_gids, save_name):
    if Path(save_name).exists():
        return torch.load(save_name)
    gids = []
    chapter_orders = []
    encodings = []
    labels = []
    for gid in tqdm(all_gids):
        chapter_order, encoding, lab = generate_examples("guten_roberta_tokens/{}.pkl".format(gid))
        gids.extend([gid]*len(lab))
        chapter_orders.extend(chapter_order)
        encodings.extend(encoding)
        labels.extend(lab)
    dataset = ChapterBoundaryDataset(gids, chapter_orders, encodings, labels)
    torch.save(dataset, save_name)
    return dataset

with open("train_gids.txt") as f:
    train_gids = [x.strip() for x in f.readlines()]
with open("eval_gids.txt") as f:
    val_gids = [x.strip() for x in f.readlines()]
with open("test_gids.txt") as f:
    test_gids = [x.strip() for x in f.readlines()]
print(len(train_gids), len(val_gids), len(test_gids))

train_dataset = create_datasets(train_gids, "train_dataset")
val_dataset = create_datasets(val_gids, "val_dataset")
test_dataset = create_datasets(test_gids, "test_dataset")

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

model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()
trainer.save_model('chapter_order_model')
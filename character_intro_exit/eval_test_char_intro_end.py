import pickle
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments

rand_seed = 1729
torch.manual_seed(rand_seed)
np.random.seed(rand_seed)

class CharacterDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def parse_char_centered_toks(tokens):
    CHAR_TOKEN = 50265
    TOKEN_WINDOW = 510
    idx = len(tokens) // 2
    lidx = idx
    while lidx >= 0 and tokens[lidx] != CHAR_TOKEN:
        lidx -= 1
    ridx = idx
    while ridx < len(tokens) and tokens[ridx] != CHAR_TOKEN:
        ridx += 1
    if idx - lidx <= ridx - idx:
        idx = lidx
    else:
        idx = ridx
    left = max(0, idx - (TOKEN_WINDOW//2))
    right = min(len(tokens), idx + (TOKEN_WINDOW//2))
    if right - left < TOKEN_WINDOW:
        if left == 0:
            right += (TOKEN_WINDOW - right - left)
        else:
            left -= (TOKEN_WINDOW - right - left)
    left = max(0, left)
    right = min(len(tokens), right)
    subtoks = [0] + tokens[left:right] + [2]
    pad = TOKEN_WINDOW + 2 - len(subtoks)
    return (
        subtoks + [1]*pad, 
        [1] * len(subtoks) + [0] * pad
    )
    

def generate_examples(pkl_path):
    with open(pkl_path, "rb") as f:
        result = pickle.load(f)
    start = []
    middle = []
    end = []
    for charid in result:
        for idx, vals in enumerate(result[charid]):
            if idx == 0:
                start.append(vals)
            elif idx == len(result[charid]) - 1:
                end.append(vals)
            else:
                middle.append(vals)
    labels = [0]*len(start) + [1]*len(middle) + [2]*len(end)
    input_ids = []
    attention_masks = []
    for _, _, _, subtoks in start + middle + end:
        input_id, attention_mask = parse_char_centered_toks(subtoks)
        input_ids.append(input_id)
        attention_masks.append(attention_mask)
    return input_ids, attention_masks, labels
    
test_dataset = torch.load("test_test_dataset")

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

model = RobertaForSequenceClassification.from_pretrained('character_intro_end_model')
model.eval()

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
)

def compute_metrics(preds, labels):
    preds = preds.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
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
print(compute_metrics(results.predictions, results.label_ids))
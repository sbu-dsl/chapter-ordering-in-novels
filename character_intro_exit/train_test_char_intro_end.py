import pickle
import torch
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments

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
    
with open("train_gids.txt") as f:
    train_gids = [x.strip() for x in f.readlines()]
with open("eval_gids.txt") as f:
    val_gids = [x.strip() for x in f.readlines()]
with open("test_gids.txt") as f:
    test_gids = [x.strip() for x in f.readlines()]
print(len(train_gids), len(val_gids), len(test_gids))

def create_datasets(gids, save_name):
    # if Path(save_name).exists():
    #     return torch.load(save_name)
    print(save_name)
    input_ids = []
    attention_masks = []
    labels = []
    for gid in tqdm(gids):
        input_id, attention_mask, label = generate_examples("test-character-intros-ends/{}.pkl".format(gid))
        input_ids.extend(input_id)
        attention_masks.extend(attention_mask)
        labels.extend(label)
    batch = {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_masks)
    }
    dataset = CharacterDataset(batch, labels)
    torch.save(dataset, save_name)
    return dataset

train_dataset = create_datasets(train_gids, "test_train_dataset")
val_dataset = create_datasets(val_gids, "test_val_dataset")
test_dataset = create_datasets(test_gids, "test_test_dataset")

training_args = TrainingArguments(
    output_dir='./test_results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    learning_rate=1e-5,
    logging_dir='./test_logs',            # directory for storing logs
    save_steps=1000,
    logging_steps=10,
)

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)
special_tokens_dict = {'additional_special_tokens': ['<main>', '<other>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
model.resize_token_embeddings(len(tokenizer))
model.train()

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()
trainer.save_model('test_character_intro_end_model')
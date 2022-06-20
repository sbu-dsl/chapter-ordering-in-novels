import csv
import torch
import numpy as np
from tqdm import tqdm

rand_seed = 1729
torch.manual_seed(rand_seed)
np.random.seed(rand_seed)

class ChapterBoundaryDataset(torch.utils.data.Dataset):
    def __init__(self, gids, chapter, encodings, labels):
        self.gids = gids
        self.chapter = chapter
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = self.encodings[idx]
        item['labels'] = torch.tensor(self.labels[idx])
        item['gid'] = self.gids[idx]
        item['chapter'] = self.chapter[idx]
        return item

    def __len__(self):
        return len(self.labels)

print("Loading test dataset")
test_dataset = torch.load("test_dataset")
print(len(test_dataset))


results = torch.load('results.pt')
logits = results.predictions
print(logits.shape)
with open("first_chapter_predictions.csv", "w") as csvfile:
    headers = ["gid", "chapter", "score0", "score1"]
    writer = csv.DictWriter(csvfile, headers)
    writer.writeheader()
    for idx, d in tqdm(enumerate(test_dataset)):
        gid = d['gid']
        chapter = d['chapter']
        leftlogit, rightlogit = logits[idx]
        writer.writerow({
            "gid": gid,
            "chapter": chapter,
            "score0": leftlogit,
            "score1": rightlogit
        })

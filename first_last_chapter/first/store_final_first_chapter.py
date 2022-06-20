import csv
import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path
from scipy.special import softmax
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

rand_seed = 1729
np.random.seed(rand_seed)

def compute_metrics(preds, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

book_to_preds = defaultdict(list)
with open("first_chapter_predictions.csv", "r") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in tqdm(reader):
        gid = row['gid']
        chapter = int(row['chapter'])
        score0 = float(row['score0'])
        score1 = float(row['score1'])
        l, r = softmax([score0, score1])
        book_to_preds[gid].append((r, chapter))

p = Path("test_gids_ordering_data")
for gid in book_to_preds:
    first_chapter_preds = book_to_preds[gid]
    with open(p / gid / "first_chapter_probs.pkl", "wb") as f:
        pickle.dump(first_chapter_preds, f, pickle.HIGHEST_PROTOCOL)
        
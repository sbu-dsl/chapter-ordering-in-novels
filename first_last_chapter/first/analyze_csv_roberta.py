import csv
import numpy as np
from tqdm import tqdm
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

maxCheck = 5
all_scores = [0 for _ in range(maxCheck)]
total = 0
for gid in book_to_preds:
    total += 1
    preds = book_to_preds[gid]
    top_chapters = [x[1] for x in sorted(preds, reverse=True)]
    for idx, chapnum in enumerate(top_chapters[:maxCheck]):
        if chapnum == 0:
            for j in range(idx, maxCheck):
                all_scores[j] += 1
for score in all_scores:
    print(score, total, score / total)

    

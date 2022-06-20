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
book_to_labels = defaultdict(list)
all_preds = []
all_labels = []
with open("test_chapter_orders.csv", "r") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in tqdm(reader):
        gid = row['gid']
        left_chapter = int(row['left_chapter'])
        right_chapter = int(row['right_chapter'])
        left_score = float(row['left_score'])
        right_score = float(row['right_score'])
        if left_chapter == right_chapter - 1:
            book_to_labels[gid].append(1)
            all_labels.append(1)
        else:
            book_to_labels[gid].append(0)
            all_labels.append(0)
        l, r = softmax([left_score, right_score])
        if l > 0.25:
            book_to_preds[gid].append(0)
            all_preds.append(0)
        else:
            book_to_preds[gid].append(1)
            all_preds.append(1)
        

all_scores = [[] for _ in range(5)]
for gid in book_to_preds:
    preds = book_to_preds[gid]
    labels = book_to_labels[gid]
    scores = compute_metrics(np.array(preds), np.array(labels))
    all_scores[0].append(scores['accuracy'])
    all_scores[1].append(scores['f1'])
    all_scores[2].append(scores['precision'])
    all_scores[3].append(scores['recall'])
    all_scores[4].append(len(labels))
for i in range(5):
    print(np.mean(all_scores[i]))

print(compute_metrics(all_preds, all_labels), len(all_labels))
    

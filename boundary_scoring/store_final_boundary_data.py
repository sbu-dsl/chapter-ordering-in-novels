import csv
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.special import softmax
from collections import defaultdict


np.random.seed(42)

with open("test_gids.txt") as f:
    test_gids = [x.strip() for x in f.readlines()]


book_to_preds = defaultdict(dict)
with open("test_chapter_orders.csv", "r") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in tqdm(reader):
        gid = row['gid']
        left_chapter = int(row['left_chapter'])
        right_chapter = int(row['right_chapter'])
        left_score = float(row['left_score'])
        right_score = float(row['right_score'])
        l, r = softmax([left_score, right_score])
        book_to_preds[gid]["{}-{}".format(left_chapter, right_chapter)] = r

p = Path("test_gids_ordering_data")
for gid in book_to_preds:
    gpath = p / gid
    if not gpath.exists():
        gpath.mkdir()
    pairs_to_probs = book_to_preds[gid]
    max_chap = 0
    for key in pairs_to_probs:
        lnum, rnum = map(int, key.split('-'))
        max_chap = max(max_chap, lnum, rnum)
    all_chap_scores = []
    for i in range(max_chap+1):
        chap_scores = []
        for j in range(max_chap+1):
            if i == j:
                chap_scores.append(0)
            else:
                chap_scores.append(pairs_to_probs["{}-{}".format(i,j)])
        all_chap_scores.append(chap_scores)
    with open(gpath / "boundary_probs.pkl", "wb") as f:
        pickle.dump(all_chap_scores, f, pickle.HIGHEST_PROTOCOL)
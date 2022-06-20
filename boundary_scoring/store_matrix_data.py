import pickle
import csv
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

print("Writing dist matrix")
matrix_path = Path("boundary-matrix-data")
if not matrix_path.exists():
    matrix_path.mkdir()
guten_token_path = Path("tokens")
for gid in tqdm(test_gids):
    with open(guten_token_path / "{}.pkl".format(gid), "rb") as f:
        num_chapters = len(pickle.load(f))
    random_order = np.arange(num_chapters)
    np.random.shuffle(random_order)
    # 3, 2, 1, 0, 4
    cnum_to_rand_cnum = {}
    for idx, cnum in enumerate(random_order):
        cnum_to_rand_cnum[cnum] = idx

    pairs_to_probs = book_to_preds[gid]
    all_chap_scores = []
    for i in range(num_chapters):
        chap_scores = []
        for j in range(num_chapters):
            if i == j:
                chap_scores.append(0)
            else:
                chap_scores.append(pairs_to_probs["{}-{}".format(i,j)])
        scaled_scores = np.array(chap_scores) / max(chap_scores)
        all_chap_scores.append(scaled_scores)
    
    dist_mat = [[0]*num_chapters for _ in range(num_chapters)]
    for i in range(num_chapters):
        for j in range(num_chapters):
            score = all_chap_scores[i][j]
            x = cnum_to_rand_cnum[i]
            y = cnum_to_rand_cnum[j]
            dist_mat[x][y] = score
    with open(matrix_path / "{}.pkl".format(gid), "wb") as f:
        pickle.dump((random_order, dist_mat), f, pickle.HIGHEST_PROTOCOL)


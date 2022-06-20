import pickle
from tqdm import tqdm
from pathlib import Path
import torch
import numpy as np
from scipy.special import softmax

np.random.seed(42)

with open("test_gids.txt") as f:
    test_gids = [x.strip() for x in f.readlines()]

def generate_examples(pkl_path):
    with open(pkl_path, "rb") as f:
        result = pickle.load(f)
    start = []
    middle = []
    middle_lens = []
    end = []
    num_chars = 0
    for charid in result:
        middle_cnt = 0
        for idx, vals in enumerate(result[charid]):
            if idx == 0:
                start.append(vals)
            elif idx == len(result[charid]) - 1:
                end.append(vals)
            else:
                middle.append(vals)
                middle_cnt += 1
        middle_lens.append(middle_cnt)
        num_chars += 1
    labels = [0]*len(start) + [1]*len(middle) + [2]*len(end)
    cnums = []
    for cnum, _, _, _ in start + middle + end:
        cnums.append(cnum)
    return cnums, middle_lens, num_chars, labels

def save_char_intro_end_meta(gids):
    p = Path("char_intro_end_meta.pkl")
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    book_infos = []
    all_labels = []
    for gid in tqdm(gids):
        cnums, middlelens, num_chars, labels = generate_examples("character-intros-ends/{}.pkl".format(gid))
        book_infos.append((gid, cnums, middlelens, num_chars))
        all_labels.extend(labels)
    with open(p, "wb") as f:
        pickle.dump((book_infos, all_labels), f, pickle.HIGHEST_PROTOCOL)
    return book_infos, all_labels

results = torch.load('results.pt')
labels = softmax(results.predictions,axis=1)
book_infos, _ = save_char_intro_end_meta(test_gids)

label_idx = 0
gid_to_char_chapters = {}
for gid, cnums, middlelens, num_chars in book_infos:
    book_labels = labels[label_idx:label_idx+len(cnums)]
    label_idx += len(cnums)
    total_middle_len = sum(middlelens)
    all_char_chapters = []
    char_middles = 0
    for cidx in range(num_chars):
        middle = middlelens[cidx]
        
        intro_chap = cnums[cidx]
        intro_label = book_labels[cidx]

        middle_base = num_chars+char_middles
        char_middles += middle
        middle_chaps = cnums[middle_base:middle_base + middle]
        middle_labels = book_labels[middle_base:middle_base + middle]

        end_chap = cnums[num_chars + total_middle_len + cidx]
        end_label = book_labels[num_chars + total_middle_len + cidx]
        
        char_chapters = [intro_chap] + middle_chaps + [end_chap]
        char_labels = np.concatenate(([intro_label],middle_labels,[end_label]),axis=0)
        # print(char_chapters, char_labels)
        intro_prob, intro_chap = sorted(zip([p[0] for p in char_labels], char_chapters), reverse=True)[0]
        end_prob, end_chap = sorted(zip([p[2] for p in char_labels], char_chapters), reverse=True)[0]
        # print(intro_prob, intro_chap)
        # print(end_prob, end_chap)
        remaining_chaps = set(char_chapters)
        remaining_chaps -= {intro_chap, end_chap}
        # print(remaining_chaps)
        all_char_chapters.append((intro_chap, remaining_chaps, end_chap))
    gid_to_char_chapters[gid] = all_char_chapters

print("Writing dist matrix")
matrix_path = Path("character-matrix-data")
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

    dist_mat = [[0.5]*num_chapters for _ in range(num_chapters)]
    for i in range(num_chapters):
        dist_mat[i][i] = 0
    char_chapters = gid_to_char_chapters[gid]
    counts = [[0]*num_chapters for _ in range(num_chapters)]
    for start_chap, middle_chaps, end_chap in char_chapters:
        for chap in middle_chaps.union({end_chap}):
            counts[start_chap][chap] += 1
        for chap in middle_chaps.union({start_chap}):
            counts[chap][end_chap] += 1

    for i in range(num_chapters):
        for j in range(i+1, num_chapters):
            score = 0.5
            if counts[i][j] > 0 or counts[j][i] > 0:
                score = counts[i][j] / (counts[i][j] + counts[j][i])
            x = cnum_to_rand_cnum[i]
            y = cnum_to_rand_cnum[j]
            dist_mat[x][y] = score
            dist_mat[y][x] = 1 - score

    
    with open(matrix_path / "{}.pkl".format(gid), "wb") as f:
        pickle.dump((random_order, dist_mat), f, pickle.HIGHEST_PROTOCOL)


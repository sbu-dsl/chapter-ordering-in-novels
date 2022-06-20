import pickle
from tqdm import tqdm
from pathlib import Path
import torch
import numpy as np
from collections import defaultdict
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
# labels = results.predictions.argmax(-1)
labels = softmax(results.predictions,axis=1)
book_infos, _ = save_char_intro_end_meta(test_gids)

label_idx = 0
for gid, cnums, middlelens, num_chars in tqdm(book_infos):
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
        char_predictions = list(zip(char_chapters, char_labels))
        all_char_chapters.append(char_predictions)
    p = Path("test_gids_ordering_data")
    gpath = p / gid
    if not gpath.exists():
        gpath.mkdir()
    with open(gpath / "character_order_probs.pkl", "wb") as f:
        pickle.dump(all_char_chapters, f, pickle.HIGHEST_PROTOCOL)


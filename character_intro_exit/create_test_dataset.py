import pickle
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from lxml import etree
from transformers import RobertaTokenizerFast
from collections import defaultdict
from multiprocessing import Pool

rand_seed = 1729
torch.manual_seed(rand_seed)
np.random.seed(rand_seed)

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)
special_tokens_dict = {'additional_special_tokens': ['<main>', '<other>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

parser = etree.XMLParser(huge_tree=True, remove_blank_text=True)
TOKEN_WINDOW_LEN = 255

def pickle_char_idxs(p):
    with open(p, "rb") as f:
        tokens = pickle.load(f)
    char_idxs = defaultdict(list)
    num_chapters = 0
    for i, chapter in enumerate(tokens):
        for j, tok in enumerate(chapter):
            if "<char" in tok:
                char_idxs[tok].append((i,j))
        num_chapters += 1

    full_char_data = {}
    for char in char_idxs:
        if char_idxs[char][0][0] == char_idxs[char][-1][0]:
            continue
        chapter_char_idxs = [[] for _ in range(num_chapters)]
        for chap_num, tok_num in char_idxs[char]:
            chapter_char_idxs[chap_num].append(tok_num)
        
        char_data = []
        for chap_num, chapter in enumerate(chapter_char_idxs):
            idxs = []
            if len(chapter) == 1:
                idxs = [0]
            elif len(chapter) >= 2:
                idxs = [0, len(chapter)-1]
            for idx in idxs:
                tok_num = chapter[idx]
                lidx = max(0, tok_num - TOKEN_WINDOW_LEN)
                ridx = min(len(tokens[chap_num]), tok_num + TOKEN_WINDOW_LEN + 1)
                window = tokens[chap_num][lidx:ridx]
                for i in range(len(window)):
                    if window[i] == char:
                        window[i] = "<main>"
                    elif "<char" in window[i]:
                        window[i] = "<other>"            
                tokenized = tokenizer(window, add_special_tokens=False, is_split_into_words=True)["input_ids"]
                char_data.append((chap_num, tok_num-lidx, window, tokenized))
        full_char_data[char] = char_data

    with open("character-intros-ends/{}.pkl".format(p.stem), "wb") as f:
        pickle.dump(full_char_data, f, pickle.HIGHEST_PROTOCOL)

all_gid_path = Path("good_gids.txt")
with open(all_gid_path) as f:
    all_gids = [x.strip() for x in f.readlines()]

guten_xml_path = Path("tokens")
all_paths = [guten_xml_path / "{}.pkl".format(gid) for gid in all_gids]
with Pool(24) as pool:
    r = list(tqdm(pool.imap(pickle_char_idxs, all_paths), total=len(all_paths)))

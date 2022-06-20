import pickle
import random
from lxml import etree
from pathlib import Path
from tqdm import tqdm
from transformers import RobertaTokenizerFast
from collections import defaultdict

random.seed(42)

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)
special_tokens_dict = {'additional_special_tokens': ['<main>', '<other>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

all_gid_path = Path("good_guten_ids.txt")
with open(all_gid_path) as f:
    all_gids = [x.strip() for x in f.readlines()]

parser = etree.XMLParser(huge_tree=True, remove_blank_text=True)
guten_xml_path = Path("tokens")
TOKEN_WINDOW_LEN = 255
NUM_SAMPLES = 5
for p in tqdm(list(guten_xml_path.glob("*.pkl"))):
    with open(p, "rb") as f:
        tokens = pickle.load(f)
    char_idxs = defaultdict(list)
    for i, chapter in enumerate(tokens):
        for j, tok in enumerate(chapter):
            if "<char" in tok:
                char_idxs[tok].append((i,j))
    full_char_data = {}
    for char in char_idxs:
        char_data = []
        if char_idxs[char][0][0] == char_idxs[char][-1][0]:
            continue
        idxs = [0] + random.sample(range(1,len(char_idxs[char])-1), min(len(char_idxs[char])-2, NUM_SAMPLES)) + [len(char_idxs[char])-1]
        
        for i in idxs:
            chap_num, tok_num = char_idxs[char][i]
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
    
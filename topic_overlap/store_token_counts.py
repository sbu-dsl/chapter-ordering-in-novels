import pickle
import numpy as np
from lxml import etree
from tqdm import tqdm
from pathlib import Path
from collections import Counter
import spacy

#loading the english language small model of spacy
en = spacy.load('en_core_web_sm')
stopwords = en.Defaults.stop_words
# print(stopwords)

np.random.seed(42)

with open("test_gids.txt") as f:
    test_gids = [x.strip() for x in f.readlines()]

parser = etree.XMLParser(huge_tree=True, remove_blank_text=True)
xml_dir = Path("good_gids_xmls")
for gid in tqdm(test_gids):
    xml_path = xml_dir / "{}".format(gid) / "character_coref_annotated.xml"
    root = etree.parse(str(xml_path), parser=parser)
    body = root.find("body")
    all_lemmas = []
    for header in body.iter("header"):
        lemmas = []
        for t in header.iter("t"):
            lemma = t.get("lemma")
            if not lemma.isalnum() or lemma.lower() in stopwords:
                continue
            lemmas.append(lemma)
        all_lemmas.append(lemmas)

    # all_lemmas = [["a","a","b","c"], ["a","b","c","d","e"], ["a","a"]]
    num_chapters = len(all_lemmas)
    random_order = np.arange(num_chapters)
    np.random.shuffle(random_order)
    cnum_to_rand_cnum = {}
    for idx, cnum in enumerate(random_order):
        cnum_to_rand_cnum[cnum] = idx

    chapter_lens = [len(x) for x in all_lemmas]
    chapter_counts = []
    chapter_weights = [[0.0]*num_chapters for _ in range(num_chapters)]
    for chapter in all_lemmas:
        counts = Counter(chapter)
        chapter_counts.append(counts)
    for i, chapter_count in enumerate(chapter_counts):
        for j, other_chapter_count in enumerate(chapter_counts[i+1:], start=i+1):
            weight = 0
            if chapter_lens[i] <= chapter_lens[j]:
                for word in chapter_count:
                    weight += min(chapter_count[word], other_chapter_count[word])
                weight /= chapter_lens[i]
            else:
                for word in other_chapter_count:
                    weight += min(chapter_count[word], other_chapter_count[word])
                weight /= chapter_lens[j]
            x = cnum_to_rand_cnum[i]
            y = cnum_to_rand_cnum[j]
            chapter_weights[x][y] = chapter_weights[y][x] = weight
    # print(chapter_weights)
    # for row in chapter_weights:
    #     print([round(x,2) for x in row])
    # print(random_order, gid)
    # break
    with open("overlap-matrix-data/{}.pkl".format(gid), "wb") as f:
        pickle.dump((random_order,chapter_weights), f, pickle.HIGHEST_PROTOCOL)

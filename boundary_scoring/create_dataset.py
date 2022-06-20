import pickle
from lxml import etree
from pathlib import Path
from transformers import RobertaTokenizerFast
from tqdm import tqdm

all_gid_path = Path("good_guten_ids.txt")
with open(all_gid_path) as f:
    all_gids = [x.strip() for x in f.readlines()]

data_path = Path("ordering-data")
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
parser = etree.XMLParser(huge_tree=True, remove_blank_text=True)
guten_xml_path = Path("good_gid_xmls")
for gid in tqdm(all_gids):
    xml_path = guten_xml_path / gid / "header_annotated.xml"
    root = etree.parse(str(xml_path), parser=parser)
    body = root.find("body")
    with open(data_path / "{}.pkl".format(gid), "wb") as f:
        chapters = []
        for header in body.iter("header"):
            tokens = []
            for p in header.iter("p"):
                tokenized = tokenizer(p.text, add_special_tokens=False)["input_ids"]
                tokens.append(tokenized)
            chapters.append(tokens)
        pickle.dump(chapters, f, pickle.HIGHEST_PROTOCOL)
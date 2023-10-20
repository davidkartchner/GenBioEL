import os
import sys
import numpy as np
from tqdm import tqdm
import pickle
import json
import argparse
import pickle
import pandas as pd
from transformers import BartTokenizer, AutoTokenizer
from trie import Trie

sys.setrecursionlimit(8000)

parser = argparse.ArgumentParser()
# parser.add_argument('--target_kb_path', type=str, help='Path to target KB')
# parser.add_argument('--output_trie_path', type=str, help='Path to put output trie')
parser.add_argument(
    "--data_dir",
    type=str,
    help="Directory with target_kb.json.  trie.pkl will also be output to this path.",
)
parser.add_argument(
    "--use_biobart",
    action="store_true",
    help="Use BioBART tokenizer instead of regular BART tokenizer",
)
args = parser.parse_args()

bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
biobart_tokenizer = AutoTokenizer.from_pretrained("GanjinZero/biobart-v2-large")

if args.use_biobart:
    tokenizer = biobart_tokenizer
else:
    tokenizer = bart_tokenizer

with open(f"{args.data_dir}/target_kb.json", "r") as f:
    cui2str = json.load(f)

entities = []
for cui in cui2str:
    entities += cui2str[cui]


trie = Trie(
    [16] + list(tokenizer(" " + entity.lower())["input_ids"][1:])
    for entity in tqdm(entities)
).trie_dict

if args.use_biobart:
    with open(f"{args.data_dir}/biobart_trie.pkl", "wb") as w_f:
        pickle.dump(trie, w_f)
else:
    with open(f"{args.data_dir}/trie.pkl", "wb") as w_f:
        pickle.dump(trie, w_f)


print("finish running!")

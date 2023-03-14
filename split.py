import json
import argparse
import random
import glob
import os
import gzip
import tqdm
import numpy as np

def split(file_path: str, out_path: str):
    random.seed(43)
    total_size = 0
    alldata_path = os.path.join(out_path, "alldata.jsonl.gz")

    with gzip.open(alldata_path, "rt", encoding="utf-8") as f:
        lines = f.readlines()
    
    indices = list(range(len(lines)))
    random.shuffle(indices)
    train_split = int(len(indices) * 0.9)
    valid_split = int(len(indices) * 0.95)

    train_path = os.path.join(out_path, "train.jsonl.gz")
    with gzip.open(train_path, "wt", encoding="utf-8") as f:
        for i in tqdm.tqdm(indices[:train_split]):
            f.write(lines[i])
    
    valid_path = os.path.join(out_path, "valid.jsonl.gz")
    with gzip.open(valid_path, "wt", encoding="utf-8") as f:
        for i in tqdm.tqdm(indices[train_split:valid_split]):
            f.write(lines[i])
    
    test_path = os.path.join(out_path, "test.jsonl.gz")
    with gzip.open(test_path, "wt", encoding="utf-8") as f:
        for i in tqdm.tqdm(indices[valid_split:]):
            f.write(lines[i])


if __name__ == "__main__":
    split("../MewNet/dataset", "./dataset")
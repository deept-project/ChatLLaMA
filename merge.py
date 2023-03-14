import json
import argparse
import random
import glob
import os
import gzip
import tqdm
import numpy as np

def merge(file_path: str, out_path: str):
    random.seed(43)
    total_size = 0
    alldata_path = os.path.join(out_path, "alldata.jsonl.gz")

    jsonl_files = glob.glob(os.path.join(file_path, "*.jsonl"))
    with gzip.open(alldata_path, "wt", encoding="utf-8") as fout:
        for jsonl_file in jsonl_files:
            print(jsonl_file)
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in tqdm.tqdm(f):
                    fout.write(line)
                    total_size += 1

if __name__ == "__main__":
    merge("../MewNet/dataset", "./dataset")
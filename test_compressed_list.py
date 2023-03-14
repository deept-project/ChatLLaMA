import os
import sys
from llama.data.compressed_list import clist
import tqdm
import random
import json
import gzip

alldata_path = os.path.join("./dataset", "alldata.jsonl.gz")

with gzip.open(alldata_path, "rt", encoding="utf-8") as f:
    lines = f.readlines()

origin_list = lines

print(sys.getsizeof(origin_list))

lst = clist()
for i in tqdm.tqdm(range(len(origin_list)), total=len(origin_list)):
    lst.append(origin_list[i])

print(sys.getsizeof(lst))

new_list = []
for item in lst:
    new_list.append(item)

print(new_list == origin_list)
# print(new_list, origin_list)
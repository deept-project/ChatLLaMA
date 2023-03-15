
import math
import time
import os
import random
import json
from typing import Optional
import numpy as np

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

import gzip
import tqdm
import random


class JsonlGzipDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, path: str) -> None:
        self.tokenizer = tokenizer
        self.path = path
        self.data = []
        with gzip.open(self.path, "rt", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.data.append(obj["text"])

    def __getitem__(self, index):
        text = self.data[index]
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return token_ids

    def __len__(self):
        return len(self.data)
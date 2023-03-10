
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

import tqdm
import random


class JsonlDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, dir_path: str, source: str) -> None:
        self.tokenizer = tokenizer
        self.dir_path = dir_path
        self.source = source
        self.data = []
        with open(self.dir_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.data.append(obj["text"])

    def __getitem__(self, index):
        text = self.data[index]
        token_ids = self.tokenizer.encode(text)
        return token_ids

    def __len__(self):
        return len(self.data)
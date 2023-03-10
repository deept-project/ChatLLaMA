import json

from tokenizers import Tokenizer
from tokenizers.models import BPE
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

from tokenizers.trainers import BpeTrainer
trainer = BpeTrainer(
    vocab_size=32000,
    min_frequency=0,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[BOS]", "[EOS]"]
)

from tokenizers.pre_tokenizers import Whitespace
tokenizer.pre_tokenizer = Whitespace()

import os
training_dataset = []
total_line = 0
for root, dirs, files in os.walk("../MewNet/dataset"):
    for file in files:
        file_path = os.path.join(root, file)
        if file.endswith('.jsonl') and "train" not in file and "valid" not in file and "test" not in file:
            training_dataset.append(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                total_line += len(f.readlines())
            print(file_path)

def training_dataset_iterator():
    for dataset_path in training_dataset:
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)['text']

# tokenizer.train(training_files, trainer)
tokenizer.train_from_iterator(training_dataset_iterator(), trainer, total_line)


tokenizer.save("tokenizer-mewnet.json")
# tokenizer.save_pretrained("code-search-net-tokenizer")

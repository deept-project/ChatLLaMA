import json

from transformers import AutoTokenizer
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

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
    for dataset_path in [training_dataset[3]]:
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)['text']

# tokenizer.train(training_files, trainer)
tokenizer = old_tokenizer.train_new_from_iterator(training_dataset_iterator(), 52000, total_line)

tokenizer.save_pretrained("tokenizer-mewnet")

import os
import json
import glob
import argparse

import tqdm

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from llama.lightning.llama import PretrainLLaMA
from llama.model.configuration_llama import LLaMAConfig
from llama.model.tokenization_llama_fast import LLaMATokenizerFast

from llama.data.dataset.jsonl_dataset import JsonlDataset
from llama.data.dataset.jsonl_gzip_dataset import JsonlGzipDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler

import lightning_fabric

from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./checkpoints/LLaMA-base/config.json", help='JSON file for configuration')
    parser.add_argument('-a', '--accelerator', type=str, default="gpu", help='training device')
    parser.add_argument('-d', '--device', type=str, default="3", help='training device ids')
    parser.add_argument('-s', '--seed', type=int, default=43, help='training seed')
    parser.add_argument('-b', '--batch-size', type=int, default=4, help='training seed')
    parser.add_argument('-cp', '--checkpoint', type=str, default="checkpoints/LLaMA-base", help='checkpoint path')
    args = parser.parse_args()

    hparams = LLaMAConfig.from_json_file(args.config)

    lightning_fabric.utilities.seed.seed_everything(args.seed)

    tokenizer = LLaMATokenizerFast.from_pretrained(args.checkpoint)

    train_dataset = JsonlGzipDataset(tokenizer, "./dataset/test.jsonl.gz")
    valid_dataset = JsonlGzipDataset(tokenizer, "./dataset/test.jsonl.gz")
        
    collate_fn = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=16, shuffle=True, pin_memory=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=16, shuffle=False, pin_memory=True, collate_fn=collate_fn)

    model = PretrainLLaMA(hparams)
    checkpoint_callback = ModelCheckpoint(dirpath=None, save_last=True, every_n_train_steps=2000)

    devices = [int(n.strip()) for n in args.device.split(",")]
    trainer_params = {
        "accelerator": args.accelerator,
        "callbacks": [checkpoint_callback],
    }

    if args.accelerator != "cpu":
        trainer_params["devices"] = devices

    if len(devices) > 1:
        trainer_params["strategy"] = "ddp"

    trainer_params.update(hparams.trainer)

    if hparams.train.fp16_run:
        trainer_params["amp_backend"] = "native"
        trainer_params["precision"] = 16
    
    # profiler = AdvancedProfiler(filename="profile.txt")
    
    trainer = pl.Trainer(**trainer_params) # , profiler=profiler, max_steps=200
    # resume training
    ckpt_path = None
    if os.path.exists("logs/lightning_logs"):
        versions = glob.glob("logs/lightning_logs/version_*")
        if len(list(versions)) > 0:
            last_ver = sorted(list(versions))[-1]
            last_ckpt = os.path.join(last_ver, "checkpoints/last.ckpt")
            if os.path.exists(last_ckpt):
                ckpt_path = last_ckpt
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader, ckpt_path=ckpt_path)

if __name__ == "__main__":
  main()
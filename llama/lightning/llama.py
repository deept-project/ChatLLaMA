


from typing import Dict
import torch
from torch import nn, optim
from torch.nn import functional as F

import pytorch_lightning as pl

from ..model.configuration_llama import LLaMAConfig

from ..model.modeling_llama import LLaMAModel, LLaMALMHeadModel

class PretrainLLaMA(pl.LightningModule):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        self.lm_model = LLaMALMHeadModel()
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=True)

    def forward(self, source_tokens: Dict[str, torch.Tensor], target_tokens: Dict[str, torch.Tensor]):
        inputs, labels = source_tokens, target_tokens

        input_ids, input_mask = inputs["token_ids"], inputs["mask"]
        label_ids, label_mask = labels["token_ids"], labels["mask"]

        batch_size = input_ids.shape[0]

        # in lightning, forward defines the prediction/inference actions
        transformer_outputs = self.lm_model(
            input_ids=input_ids,
            attention_mask=input_mask,
            decoder_input_ids=label_ids,
            decoder_attention_mask=label_mask,
            use_cache=False,
        )
        # (batch_size, sequence_length, hidden_size)
        hidden_states = transformer_outputs.last_hidden_state
        # (batch_size, sequence_length, vocab_size)
        lm_logits = self.lm_head(hidden_states)

        return lm_logits

    def training_step(self, batch, batch_idx: int):
        inputs, labels = batch

        input_ids, input_mask = inputs["token_ids"], inputs["mask"]
        label_ids, label_mask = labels["token_ids"], labels["mask"]

        batch_size = input_ids.shape[0]

        lm_logits = self.forward(
            source_tokens={
                'token_ids': input_ids,
                'mask': input_mask
            },
            target_tokens={
                'token_ids': label_ids[..., :-1],
                'mask': label_mask[..., :-1]
            }
        )

        shift_label_ids = label_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        loss = loss_fct(lm_logits.view(-1, self.vocab_size), shift_label_ids.view(-1))

        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch

        input_ids, input_mask = inputs["token_ids"], inputs["mask"]
        label_ids, label_mask = labels["token_ids"], labels["mask"]

        batch_size = input_ids.shape[0]

        lm_logits = self.forward(
            source_tokens={
                'token_ids': input_ids,
                'mask': input_mask
            },
            target_tokens={
                'token_ids': label_ids[..., :-1],
                'mask': label_mask[..., :-1]
            }
        )

        shift_label_ids = label_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        loss = loss_fct(lm_logits.view(-1, self.vocab_size), shift_label_ids.view(-1))
        self.log('val_loss', loss)

    
    def configure_optimizers(self):
        self.optim = torch.optim.AdamW(
            self.net.parameters(), 
            self.hparams.train.learning_rate, 
            betas=self.hparams.train.betas, 
            eps=self.hparams.train.eps)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=self.hparams.train.lr_decay)
        self.scheduler.last_epoch = self.current_epoch - 1

        return [self.optim], [self.scheduler]

from functools import lru_cache
from typing import Any

from pytorch_lightning import LightningModule
from torch.optim import AdamW, Optimizer
from transformers import GPT2Config, GPT2LMHeadModel, get_linear_schedule_with_warmup


class PoetLightningModel(LightningModule):
    """
    LightningModel used to train GPT2 on poetry generation.
    """

    def __init__(
        self,
        warmup_steps: int,
        vocab_size: int,
        n_positions: int,
        pretrained_name_or_path: str,
        lr: float = 1e-4,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        config = GPT2Config(vocab_size=vocab_size, n_positions=n_positions).from_pretrained(
            pretrained_name_or_path, output_hidden_states=True
        )
        model = GPT2LMHeadModel.from_pretrained(pretrained_name_or_path, config=config)
        model.resize_token_embeddings(vocab_size)

        self.model = model

    def step(self, batch: Any) -> dict:
        input_ids = batch[0]
        labels = batch[0]
        masks = batch[1]

        outputs = self.model(input_ids, labels=labels, attention_mask=masks, token_type_ids=None)

        loss = outputs[0]

        return loss

    def training_step(self, batch: Any, batch_idx: int) -> dict:
        loss = self.step(batch)

        self.log("train/loss", loss.item(), on_step=False, on_epoch=True)

    def validation_step(self, batch: Any, batch_idx: int) -> dict:
        loss = self.step(batch)

        self.log("val/loss", loss.item(), on_step=False, on_epoch=True)

    def configure_optimizers(self) -> Optimizer:
        optimizer = AdamW(params=self.model.parameters(), lr=self.hparams.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.num_training_steps,
        )

        scheduler_config = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    @lru_cache
    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if (
            isinstance(self.trainer.limit_train_batches, int)
            and self.trainer.limit_train_batches != 0
        ):
            dataset_size = self.trainer.limit_train_batches
        elif isinstance(self.trainer.limit_train_batches, float):
            # limit_train_batches is a percentage of batches
            dataset_size = len(self.trainer.datamodule.train_dataloader())
            dataset_size = int(dataset_size * self.trainer.limit_train_batches)
        else:
            dataset_size = len(self.trainer.datamodule.train_dataloader())
        num_devices = max(1, self.trainer.gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)
        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        max_estimated_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs
        if self.trainer.max_steps and 0 < self.trainer.max_steps < max_estimated_steps:
            return self.trainer.max_steps
        return max_estimated_steps

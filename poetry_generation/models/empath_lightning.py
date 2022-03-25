from typing import Any, List

import pytorch_lightning as pl
from torch.optim import AdamW, Optimizer
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from transformers import BatchEncoding, get_linear_schedule_with_warmup

from poetry_generation.models.modules.empath import EmpathModel


class EmpathLightningModel(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        lr: float,
        warmup_steps: int,
        n_classes: int = 3,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.model = EmpathModel(model_name_or_path=model_name_or_path, num_labels=n_classes)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # for logging best accuracy achieved so far
        self.val_acc_best = MaxMetric()

    def step(self, batch: BatchEncoding):
        encodings, targets = batch
        loss, logits = self.model(**encodings, labels=targets)
        return loss, logits, targets

    def training_step(self, batch: BatchEncoding):
        loss, logits, targets = self.step(batch)
        preds = logits.argmax(-1)

        # log train metrics
        acc = self.train_acc(preds, targets)

        self.log("train/loss", loss.item(), prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: BatchEncoding, batch_idx: int):
        loss, logits, targets = self.step(batch)
        preds = logits.argmax(-1)

        # log train metrics
        acc = self.val_acc(preds, targets)

        self.log("val/loss", loss.item(), prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc.reset()  # reset is necessary if you're using `num_sanity_val_steps` in trainer

        # compute and log best so far val accuracy
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, logits, targets = self.step(batch)
        preds = logits.argmax(-1)

        # log train metrics
        acc = self.val_acc(preds, targets)

        self.log("test/loss", loss.item(), prog_bar=False)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def configure_optimizers(self) -> Optimizer:
        optimizer = AdamW(params=self.model.parameters(), lr=self.hparams.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.num_training_steps,
        )

        scheduler_config = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    # @lru_cache
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

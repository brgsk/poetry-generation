from typing import Optional

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
from transformers import GPT2Tokenizer

from poetry_generation.datamodules.datasets.stanza_dataset import StanzaDataset
from poetry_generation.utils.utils import train_val_split


class StanzaDataModule(LightningDataModule):
    """
    Lightning's DataModule for poems stored as stanzas.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer_path: str,
        batch_size: int = 32,
        train_val_ratio: float = 0.8,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

    def setup(self, stage: Optional[str] = None) -> None:
        stanza_df = pd.read_csv(self.hparams.data_path)
        stanza_df = stanza_df.fillna("")

        stanza_ds = StanzaDataset(data_df=stanza_df, tokenizer=self.tokenizer)
        train_size, val_size = train_val_split(self.hparams.train_val_ratio, stanza_ds)

        self.train_ds, self.val_ds = random_split(stanza_ds, [train_size, val_size])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            sampler=RandomSampler(self.train_ds),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            sampler=SequentialSampler(self.val_ds),
        )

    def test_dataloader(self) -> DataLoader:
        pass

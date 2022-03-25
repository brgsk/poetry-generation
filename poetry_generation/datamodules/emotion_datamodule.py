
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from poetry_generation.datamodules.datasets.emotion_dataset import EmotionDataset
from poetry_generation.utils.collator import TokenCollator


class EmotionDataModule(LightningDataModule):
    """
    Lightning Datamodule implementation for storing data as EmotionDatasets.
    """

    def __init__(
        self,
        tokenizer_path: str,
        train_path: str,
        val_path: str,
        test_path: str,
        batch_size: int,
        pin_memory: bool,
    ) -> None:
        super().__init__()
        self.token_collator = TokenCollator()
        self.save_hyperparameters(logger=False)

    def train_dataloader(self) -> DataLoader:
        return self.create_dataloader(self.hparams.train_path, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self.create_dataloader(self.hparams.val_path)

    def test_dataloader(self) -> DataLoader:
        return self.create_dataloader(self.hparams.test_path)

    def create_dataloader(self, ds_path: str, shuffle=False) -> DataLoader:
        return DataLoader(
            EmotionDataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            collate_fn=TokenCollator(self.hparams.tokenizer_path),
            pin_memory=self.hparams.pin_memory,
        )

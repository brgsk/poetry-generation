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
        train_file: str,
        val_file: str,
        test_file: str,
        batch_size: int = 12,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.token_collator = TokenCollator(tokenizer_path)
        self.save_hyperparameters(logger=False)

    def train_dataloader(self) -> DataLoader:
        return self.create_dataloader(self.hparams.train_file, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self.create_dataloader(self.hparams.val_file)

    def test_dataloader(self) -> DataLoader:
        return self.create_dataloader(self.hparams.test_file)

    def create_dataloader(self, ds_path: str, shuffle=False) -> DataLoader:
        return DataLoader(
            EmotionDataset(ds_path),
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            collate_fn=TokenCollator(self.hparams.tokenizer_path),
            # collate_fn=self.token_collator,
            pin_memory=self.hparams.pin_memory,
        )

    # @staticmethod
    # def collate_fn(self):
    #     return TokenCollator(self.hparams.tokenizer_path)

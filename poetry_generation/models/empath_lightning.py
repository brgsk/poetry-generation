from typing import Any

import pytorch_lightning as pl

from poetry_generation.models.modules.emotion_classifier import RobertaEmpathModel


class EmpathLightningModel(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        learning_rate: float,
        warmup_steps: int,
        n_classes: int = 3,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.model = RobertaEmpathModel(model_name_or_path=model_name_or_path, n_classes=n_classes)

    def step(self, batch: Any):
        pass

    def training_step(self, batch: Any):
        pass

    def validation_step(self, batch: Any):
        pass

    # def test_step(self, batch: Any):
    #     pass

    def configure_optimizers(self):
        pass

import logging
import re
import warnings
from pathlib import Path
from typing import List, Sequence

import jsonlines
import pandas as pd
import pytorch_lightning as pl
import rich.syntax
import rich.tree
from fire import Fire
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from rich.logging import RichHandler


def process_poems(data_path: Path) -> None:
    poems = _loadpoems(data_path)
    texts = [poem["text"].strip() for poem in poems]
    titles = [poem["title"] for poem in poems]

    df = pd.DataFrame(list(zip(titles, texts)), columns=["title", "text"])
    stanza_title, stanza_text = _split_into_stanzas(df["title"].values, df["text"].values)
    stanza_df = pd.DataFrame(list(zip(stanza_title, stanza_text)), columns=["title", "stanza_text"])

    line_title, line_text = _split_into_lines(
        stanza_df["title"].values, stanza_df["stanza_text"].values
    )
    line_df = pd.DataFrame(list(zip(line_title, line_text)), columns=["title", "line_text"])

    line_df.to_csv("lines.csv", index=False)
    stanza_df.to_csv("stanzas.csv", index=False)


def _loadpoem(filename: Path) -> list:
    filename = str(filename)
    poems = []
    if ".jsonl" in filename:
        with jsonlines.open(filename) as reader:
            for poem in reader:
                poems.append(poem)
    return poems


def _loadpoems(path: Path):
    all_poems = []
    for file in path.iterdir():
        poems = _loadpoem(file)
        all_poems.extend(poems)
    return all_poems


def _clean_poem_text(poem_text: str) -> list:
    poem_text = poem_text.strip()
    poem_text = re.sub(" +", " ", poem_text)
    poem_text = re.sub("\n\n\r\n\r\n", "\n\r\n \n\r\n", poem_text)
    poem_text = re.sub("\n\r\n", "\n", poem_text)
    poem_text = re.sub(" +", " ", poem_text)
    return poem_text


def _split_into_stanzas(poem_title_list: list, poem_text_list: list) -> tuple[list, list]:
    return_title_list = []
    return_stanza_list = []
    for poem_index in range(len(poem_title_list)):
        poem_text = _clean_poem_text(poem_text_list[poem_index])
        poem_stanzas = poem_text.split("\n\n")
        for stanza in poem_stanzas:
            return_title_list.append(poem_title_list[poem_index])
            return_stanza_list.append(stanza)
    return return_title_list, return_stanza_list


def _split_into_lines(poem_title_list: list, poem_text_list: list) -> tuple[list, list]:
    return_title_list = []
    return_line_list = []
    for poem_index in range(len(poem_title_list)):
        poem_lines = poem_text_list[poem_index].split("\n")
        for line in poem_lines:
            return_title_list.append(poem_title_list[poem_index])
            return_line_list.append(line.strip())
    return return_title_list, return_line_list


def train_val_split(split, dataset):
    train_size = int(split * len(dataset))
    val_size = len(dataset) - train_size
    return train_size, val_size


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    FORMAT = "%(message)s"
    logging.basicConfig(level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

    logger = logging.getLogger("rich")

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - forcing debug friendly configuration
    - verifying experiment name is set when running in experiment mode
    Modifies DictConfig in place.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger(__name__)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # verify experiment name is set when running in experiment mode
    if config.get("experiment_mode") and not config.get("name"):
        log.info(
            "Running in experiment mode without the experiment name specified! "
            "Use `python run.py mode=exp name=experiment_name`"
        )
        log.info("Exiting...")
        exit()

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    # debuggers don't like GPUs and multiprocessing
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "datamodule",
        "callbacks",
        "logger",
        "test_after_training",
        "seed",
        "name",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as fp:
        rich.print(tree, file=fp)


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.
    Additionaly saves:
        - number of model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            # import wandb

            # wandb.finish()
            pass


def fire_process() -> None:
    Fire(process_poems)


if __name__ == "__main__":
    fire_process()

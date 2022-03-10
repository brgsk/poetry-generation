from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split

# from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers import PreTrainedTokenizer


class StanzaDataset(Dataset):
    def __init__(
        self,
        data_df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        max_length=512,
        padding="max_length",
    ):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []

        for i in data_df["stanza_text"].values:
            encodings_dict = tokenizer(
                "<BOS>" + i + "<EOS>", truncation=True, max_length=max_length, padding="max_length"
            )

            self.input_ids.append(torch.tensor(encodings_dict["input_ids"]))
            self.attn_masks.append(torch.tensor(encodings_dict["attention_mask"]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

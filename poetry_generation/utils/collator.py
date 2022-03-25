from typing import Any

import torch
from transformers import AutoTokenizer, BatchEncoding


class TokenCollator:
    def __init__(self, tokenizer_path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def __call__(self, batch: Any) -> BatchEncoding:
        max_sentence_len = max(len(sent) for sent in batch[0])
        encoded = self.tokenizer(
            [x for x in batch[0]],
            padding="max_length",
            max_length=max_sentence_len if max_sentence_len < 512 else 512,
            truncation=True,
            return_tensors="pt",
        )
        labels = torch.as_tensor([x for x in batch[1]])

        return encoded, labels

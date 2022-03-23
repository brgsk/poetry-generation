from typing import Any

import torch
from transformers import AutoTokenizer


class TokenizerCollator:
    def __init__(self, tokenizer_path: str, max_length: int = 512) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def __call__(self, batch: Any) -> Any:
        encoded = self.tokenizer([x[0] for x in batch])
        input_ids = torch.as_tensor([enc.input_ids for enc in encoded])
        attention_mask = torch.as_tensor([enc.attention_mask for enc in encoded])
        labels = torch.as_tensor([x[1] for x in batch])

        return (input_ids, attention_mask), labels

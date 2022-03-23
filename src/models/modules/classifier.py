from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    AutoModelWithLMHead,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)


class EmotionClassificationHead(nn.Module):
    """Head for emotion classification task."""

    def __init__(self, hidden_size: int, dropout: float, num_labels: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaEmpathModel(nn.Module):
    def __init__(
        self,
        base_model: PreTrainedModel,
        n_classes: int,
    ):
        super().__init__()
        self.roberta = base_model
        self.classifier = EmotionClassificationHead(
            hidden_size=base_model.config.hidden_size,
            dropout=base_model.config.hidden_dropout_prob,
            num_labels=n_classes,
        )
        pass

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

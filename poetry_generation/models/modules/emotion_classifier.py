from torch import nn
from transformers import AutoModel, PreTrainedModel

from poetry_generation.models.modules.emotion_classification_head import (
    EmotionClassificationHead,
)


class RobertaEmpathModel(nn.Module):
    def __init__(
        self,
        model_name_or_path: PreTrainedModel,
        n_classes: int = 3,
    ):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name_or_path)
        self.classifier = EmotionClassificationHead(
            hidden_size=self.roberta.config.hidden_size,
            dropout=self.roberta.config.hidden_dropout_prob,
            num_labels=n_classes,
        )

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
